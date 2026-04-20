from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from torch import fft

def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    
    B, C, D, H, W = x_freq.shape
    mask = th.ones((B, C, D, H, W)).to(x.device)

    cd, crow, ccol = D // 2, H // 2, W // 2
    mask[..., 
         cd - threshold:cd + threshold,
         crow - threshold:crow + threshold, 
         ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-3, -2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-3, -2, -1)).real
    
    return x_filtered


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, x_low):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, x_low)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, strides, use_conv, dims=3, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.strides = strides
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels

        x = F.interpolate(
            x, (x.shape[2] * self.strides[0], x.shape[3] * self.strides[1], x.shape[4] * self.strides[2]), mode="nearest")

        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, strides, use_conv, dims=3, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.strides = strides
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=self.strides, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=self.strides, stride=self.strides)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# class ResBlock(TimestepBlock):
#     """
#     A residual block that can optionally change the number of channels.

#     :param channels: the number of input channels.
#     :param emb_channels: the number of timestep embedding channels.
#     :param dropout: the rate of dropout.
#     :param out_channels: if specified, the number of out channels.
#     :param use_conv: if True and out_channels is specified, use a spatial
#         convolution instead of a smaller 1x1 convolution to change the
#         channels in the skip connection.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param use_checkpoint: if True, use gradient checkpointing on this module.
#     :param up: if True, use this block for upsampling.
#     :param down: if True, use this block for downsampling.
#     """

#     def __init__(
#         self,
#         channels,
#         emb_channels,
#         dropout,
#         out_channels=None,
#         use_conv=False,
#         use_scale_shift_norm=False,
#         dims=3,
#         use_checkpoint=False,
#         up=False,
#         down=False,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.emb_channels = emb_channels
#         self.dropout = dropout
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.use_checkpoint = use_checkpoint
#         self.use_scale_shift_norm = use_scale_shift_norm

#         self.in_layers = nn.Sequential(
#             normalization(channels),
#             nn.SiLU(),
#             conv_nd(dims, channels, self.out_channels, 3, padding=1),
#         )

#         self.updown = up or down

#         if up:
#             self.h_upd = Upsample(channels, False, dims)
#             self.x_upd = Upsample(channels, False, dims)
#         elif down:
#             self.h_upd = Downsample(channels, False, dims)
#             self.x_upd = Downsample(channels, False, dims)
#         else:
#             self.h_upd = self.x_upd = nn.Identity()

#         self.emb_layers = nn.Sequential(
#             nn.SiLU(),
#             linear(
#                 emb_channels,
#                 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
#             ),
#         )
#         self.out_layers = nn.Sequential(
#             normalization(self.out_channels),
#             nn.SiLU(),
#             nn.Dropout(p=dropout),
#             zero_module(
#                 conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
#             ),
#         )

#         if self.out_channels == channels:
#             self.skip_connection = nn.Identity()
#         elif use_conv:
#             self.skip_connection = conv_nd(
#                 dims, channels, self.out_channels, 3, padding=1
#             )
#         else:
#             self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

#     def forward(self, x, emb):
#         """
#         Apply the block to a Tensor, conditioned on a timestep embedding.

#         :param x: an [N x C x ...] Tensor of features.
#         :param emb: an [N x emb_channels] Tensor of timestep embeddings.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         return checkpoint(
#             self._forward, (x, emb), self.parameters(), self.use_checkpoint
#         )

#     def _forward(self, x, emb):
#         if self.updown:
#             in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
#             h = in_rest(x)
#             h = self.h_upd(h)
#             x = self.x_upd(x)
#             h = in_conv(h)
#         else:
#             h = self.in_layers(x)
            
#         emb_out = self.emb_layers(emb).type(h.dtype)

#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]

#         if self.use_scale_shift_norm:
#             out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
#             scale, shift = th.chunk(emb_out, 2, dim=1)
#             h = out_norm(h) * (1 + scale) + shift
#             h = out_rest(h)
#         else:
#             h = h + emb_out
#             h = self.out_layers(h)
#         return self.skip_connection(x) + h


class HybridCondEncoder(nn.Module):
    """
    混合条件编码器: 全局分支 + 三轴空间分支
    """
    def __init__(self, dims=3):
        super().__init__()
        
        # 分支1: 全局语义特征 (不变)
        self.global_branch = nn.Sequential(
            conv_nd(dims, 1, 32, 4, stride=4),
            nn.GroupNorm(8, 32), nn.SiLU(),
            conv_nd(dims, 32, 64, 4, stride=4),
            nn.GroupNorm(8, 64), nn.SiLU(),
            conv_nd(dims, 64, 64, 4, stride=4),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),  # [N, 64]
        )
        
        # 分支2: 三轴空间特征
        # 共享一个轻量 backbone，然后沿三个轴分别 pool
        self.spatial_backbone = nn.Sequential(
            conv_nd(dims, 1, 32, 4, stride=4),
            nn.GroupNorm(8, 32), nn.SiLU(),
            conv_nd(dims, 32, 64, 4, stride=4),
            nn.GroupNorm(8, 64), nn.SiLU(),
        )  # [N, 64, D/16, H/16, W/16]
        
        # 各轴 pool 到固定长度 K，然后 1D projection
        self.K = 4  # 每个轴保留 4 个 position
        
        self.d_pool = nn.AdaptiveAvgPool3d((self.K, 1, 1))  # [N, 64, K, 1, 1]
        self.h_pool = nn.AdaptiveAvgPool3d((1, self.K, 1))  # [N, 64, 1, K, 1]
        self.w_pool = nn.AdaptiveAvgPool3d((1, 1, self.K))  # [N, 64, 1, 1, K]
        
        # 可学习的轴重要性
        self.axis_logits = nn.Parameter(th.zeros(3))
        
        # 投影: 64 (global) + 64*K (axis, 加权合并后) = 64 + 256 = 320 → 128
        self.projection = nn.Sequential(
            nn.Linear(64 + 64 * self.K, 128),
            nn.SiLU(),
        )
    
    def forward(self, x_low):
        # Global
        global_feat = self.global_branch(x_low)  # [N, 64]
        
        # Spatial backbone (只跑一次)
        feat = self.spatial_backbone(x_low)  # [N, 64, D', H', W']
        
        # 三轴分别 pool + flatten
        d_feat = self.d_pool(feat).flatten(1)  # [N, 64*K]
        h_feat = self.h_pool(feat).flatten(1)  # [N, 64*K]
        w_feat = self.w_pool(feat).flatten(1)  # [N, 64*K]
        
        # 轴重要性加权合并
        w = F.softmax(self.axis_logits, dim=0)  # [3]
        axis_feat = w[0] * d_feat + w[1] * h_feat + w[2] * w_feat  # [N, 64*K]
        
        # 拼接 global + axis → 投影到 128
        combined = th.cat([global_feat, axis_feat], dim=1)  # [N, 64 + 256]
        cond_feat = self.projection(combined)  # [N, 128]
        
        return cond_feat
    
    def get_axis_weights(self):
        """可视化用"""
        with th.no_grad():
            return F.softmax(self.axis_logits, dim=0).cpu().numpy()


# class HybridCondEncoder(nn.Module):
#     """
#     混合条件编码器:全局分支 + Z轴空间分支
#     """
#     def __init__(self, dims=3):
#         super().__init__()
        
#         # 分支1: 全局语义特征 (用于denoising等全局任务)
#         self.global_branch = nn.Sequential(
#             conv_nd(dims, 1, 32, 4, stride=4),
#             nn.GroupNorm(8, 32), nn.SiLU(),
#             conv_nd(dims, 32, 64, 4, stride=4),
#             nn.GroupNorm(8, 64), nn.SiLU(),
#             conv_nd(dims, 64, 64, 4, stride=4),
#             nn.GroupNorm(8, 64), nn.SiLU(),
#             nn.AdaptiveAvgPool3d(1),  # [N, 64, 1, 1, 1]
#             nn.Flatten(),  # [N, 64]
#         )
        
#         # 分支2: Z轴保留的空间特征 (用于SR的方向感知)
#         # 只在XY平面降采样,保留Z轴分辨率
#         self.spatial_branch = nn.Sequential(
#             conv_nd(dims, 1, 32, kernel_size=(4, 4, 1), stride=(4, 4, 1)),
#             nn.GroupNorm(8, 32), nn.SiLU(),
#             conv_nd(dims, 32, 64, kernel_size=(4, 4, 1), stride=(4, 4, 1)),
#             nn.GroupNorm(8, 64), nn.SiLU(),
#             # 自适应池化到固定Z轴大小,比如4个slice
#             nn.AdaptiveAvgPool3d((1, 1, 4)),  # [N, 64, 1, 1, 4]
#         )
        
#         # Flatten spatial features: [N, 64*4] = [N, 256]
#         # 拼接后总维度: 64 (global) + 256 (spatial) = 320
#         # 但为了保持原来的128维接口,需要projection
#         self.projection = nn.Sequential(
#             nn.Linear(64 + 256, 128),
#             nn.SiLU(),
#         )
        
#     def forward(self, x_low):
#         # x_low: [N, 1, D, H, W]
#         global_feat = self.global_branch(x_low)  # [N, 64]
#         spatial_feat = self.spatial_branch(x_low)  # [N, 64, 1, 1, 4]

#         spatial_feat = spatial_feat.flatten(1)  # [N, 256]
        
#         # 拼接并投影
#         combined = th.cat([global_feat, spatial_feat], dim=1)  # [N, 320]
#         cond_feat = self.projection(combined)  # [N, 128]
        
#         return cond_feat


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        cond_channels=128,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=3,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.cond_channels = cond_channels
        self.dims = dims

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        # ✨ 新增：Condition Modulation Network
        # 从 x_low 中提取特征，用于调制 timestep embedding
        self.cond_modulation = nn.Sequential(
            nn.Linear(cond_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * emb_channels),
        )
        nn.init.zeros_(self.cond_modulation[-1].weight)
        nn.init.zeros_(self.cond_modulation[-1].bias)


    def forward(self, x, emb, cond_low):
        return checkpoint(
            self._forward, (x, emb, cond_low), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, cond_low):
        cond_params = self.cond_modulation(cond_low)
        gamma, beta = th.chunk(cond_params, 2, dim=1)

        # Step 2: 调制 timestep embedding
        # Emb'(t, x_c) = Emb(t) * γ(x_c) + β(x_c).  [bs, 256]
        modulated_emb = emb * (1 + gamma) + beta
        
        # 后续流程保持不变，但使用 modulated_emb 替代原始 emb
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        # ✨ 使用调制后的 embedding [bs, 64]
        emb_out = self.emb_layers(modulated_emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # 🔹 情况1：use_scale_shift_norm = True
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        # 🔹 情况2：use_scale_shift_norm = False
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        strides,
        num_res_blocks,
        channel_mult,
        attention_resolutions,
        dropout=0,
        num_heads=1,
        dims=3,
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.strides = strides
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_fp16 = use_fp16
        self.dims = dims

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)

                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, strides, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch,
            #     num_heads=num_heads,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                inch = input_block_chans.pop()
                if len(input_block_chans):
                    outch = input_block_chans.pop()
                else:
                    outch = inch
                layers = [
                    ResBlock(
                        inch * 2,
                        time_embed_dim,
                        dropout,
                        out_channels=outch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            outch,
                            num_heads=self.num_heads_upsample,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        ResBlock(
                            outch,
                            time_embed_dim,
                            dropout,
                            out_channels=outch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(outch, strides, conv_resample, dims=dims, out_channels=outch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += outch
                input_block_chans.append(outch)
        
        self.out = nn.Sequential(
            normalization(outch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

        # self.cond_encoder = nn.Sequential(
        #     conv_nd(3, 1, 32, 4, stride=4),
        #     nn.GroupNorm(8, 32), nn.SiLU(),
        #     conv_nd(3, 32, 64, 4, stride=4),
        #     nn.GroupNorm(8, 64), nn.SiLU(),
        #     conv_nd(3, 64, 128, 4, stride=4),
        #     nn.GroupNorm(8, 128), nn.SiLU(),
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(),
        # )  # 输出 [N, 128]

        self.cond_encoder = HybridCondEncoder(dims=3)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, x_low, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)

        cond_low = self.cond_encoder(x_low)

        # Encoder
        for module in self.input_blocks:
            h = module(h, emb, cond_low)
            hs.append(h)

        # Bottleneck
        h = self.middle_block(h, emb, cond_low)

        # Decoder
        for module in self.output_blocks:
            hs_ = hs.pop()
            h = th.cat([h, hs_], dim=1)
            h = module(h, emb, cond_low)
        
        h = h.type(x.dtype)

        return self.out(h)
    

class SuperResModel_noatt(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels, *args, **kwargs)

    def forward(self, x, x_low, timesteps, low_res=None, **kwargs):

        x = th.cat([x, x_low], dim=1)

        return super().forward(x, timesteps, x_low, **kwargs)
