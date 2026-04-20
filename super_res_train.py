"""
Train a super-resolution model.
"""

import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusion import dist_util, logger, create_diffusion
from diffusion.image_datasets import load_data
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import create_gaussian_diffusion
from diffusion.train_util import TrainLoop
from models.unet import SuperResModel_noatt


def ddp_setup():
    init_process_group(backend="nccl")
    print(f"Backend in use: {torch.distributed.get_backend()}")
    
    pass


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default='', help="Path to the config file")
    parser.add_argument("--resume_checkpoint",type=str, default='', help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    return parser, args


def main():
    parser, args = create_argparser()

    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    for k, v in yaml_cfg.items():
        default = parser.get_default(k)
        if hasattr(args, k) and default is not None:
            v = type(default)(v)
        setattr(args, k, v)

    ddp_setup()

    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        sigma_small=args.sigma_small,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing="")
    
    model = SuperResModel_noatt(
        in_channels=args.in_channels,
        model_channels=args.model_channels,
        out_channels=args.out_channels,
        strides=args.strides,
        num_res_blocks=args.num_res_blocks,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
        num_heads=args.num_heads,
        dims=3,
        use_scale_shift_norm=args.use_scale_shift_norm)

    logger.log('data_dir:{%s}' % args.data_dir)
    logger.log('batch_size:{%d}' % args.batch_size)
    logger.log('patch_size:{%s}' % args.patch_size)
    logger.log('type:{%s}' % args.type)
    logger.log('diffusion_steps:{%d}' % args.diffusion_steps)
    logger.log('steps:{%s}' % str(args.f_steps))
    logger.log('extra_args:{%s}' % str(args.extra_args))
    logger.log('lr:{%s}' % str(args.lr))
    logger.log('ema_rate:{%s}' % str(args.ema_rate))
    logger.log('parameters:{%d}' % (sum(param.numel() for param in model.parameters())))
    logger.log('in_channels:{%s} out_channels:{%s}' % (args.in_channels, args.out_channels))
    logger.log('model_channels:{%s}' % str(args.model_channels))
    logger.log('strides:{%s}' % str(args.strides))
    logger.log('channel_mult:{%s}' % str(model.channel_mult))
    logger.log('attention_resolutions:{%s}' % args.attention_resolutions)
    logger.log('num_res_blocks:{%s}' % str(args.num_res_blocks))
    logger.log('dropout:{%s}' % str(args.dropout))
    logger.log('num_heads:{%s}' % str(args.num_heads))
    logger.log('sigma_small:{%s}' % str(args.sigma_small))
    logger.log('predict_xstart:{%s}' % str(args.predict_xstart))

    
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        args.patch_size,
        args.num_workers
    )

    prefix = 'ckpt_' + str(args.extra_args)
    args.result_folder = os.path.join(args.result_dir, prefix)
    os.makedirs(args.result_folder, exist_ok=True)
    
    logger.log("Save results to {}".format(args.result_folder))
    logger.log("training...")
    
    TrainLoop(
        args=args,
        model=model,
        diffusion=diffusion,
        data=data,
        schedule_sampler=schedule_sampler,
    ).run_loop()


def load_superres_data(data_dir, batch_size, patch_size, num_workers):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=num_workers)

    for pet_batch in data:
        yield pet_batch


if __name__ == "__main__":
    main()
