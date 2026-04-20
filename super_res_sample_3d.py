import argparse
import os
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import blobfile as bf
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import numpy as np
import torch as th
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import time
from tqdm import tqdm


from diffusion import logger
from diffusion.image_datasets_sample import load_data
from diffusion.script_util import create_gaussian_diffusion
from models.unet import SuperResModel_noatt

from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


def ddp_setup():
    init_process_group(backend="nccl")
    pass


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default="", help="Path to the config file")
    parser.add_argument("--model_path",type=str, default="", help="Path to the model checkpoint")

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
    

    save_dir = os.path.join("Visual_results", args.model_path.split('/')[-2])
    os.makedirs(save_dir, exist_ok=True)
    logger.configure(dir=save_dir)
    logger.log("Saving sample results to {}".format(save_dir))
    
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

    
    gpu_id = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    model.load_state_dict(th.load(args.model_path))
    model = model.to(gpu_id)
    logger.log("Model loaded from {}".format(args.model_path))

    model.eval()

    dataset = load_data(
        args.sample_dir,
        args.batch_size,
        args.patch_size,
        args.num_workers)

    sample_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    num_samples = len(dataset)
    logger.log("There're {} samples in the dataset.".format(num_samples))

    all_psnr, all_ssim, all_hfen, all_msssim = [], [], [], []
    all_ssim2d = []

    with th.no_grad():
        for (j, data) in enumerate(tqdm(sample_loader, desc="Processing samples")):
            data['low_res'] = data['low_res'].squeeze(0)
            data['full_res'] = data['full_res'].squeeze(0)

            C,H,W = data['shape'][0].item(), data['shape'][1].item(), data['shape'][2].item()

            patch_num = data['low_res'].shape[0]
            data_per_gpu = patch_num // world_size
            reminder = patch_num % world_size

            low_res_chunks = th.split(data['low_res'], ([data_per_gpu+1]*(reminder)+[data_per_gpu]*(world_size-reminder)))
            full_res_chunks = th.split(data['full_res'], ([data_per_gpu+1]*(reminder)+[data_per_gpu]*(world_size-reminder)))

            low_res_data = low_res_chunks[gpu_id].to(gpu_id)
            full_res_data = full_res_chunks[gpu_id].to(gpu_id)
            
            max_patches = data_per_gpu + 1
            
            output_array = th.zeros(max_patches, *low_res_data.shape[1:]).to(gpu_id)
            low_res_array = th.zeros(max_patches, *low_res_data.shape[1:]).to(gpu_id)
            full_res_array = th.zeros(max_patches, *full_res_data.shape[1:]).to(gpu_id)

            actual_patches = low_res_data.shape[0]

            dist.barrier(device_ids=[gpu_id])

            for i in tqdm(range(actual_patches), desc="GPU {} processing patches".format(gpu_id)):
                local_low_res = low_res_data[i,:,:,:].unsqueeze(0).unsqueeze(0)
                local_full_res = full_res_data[i,:,:,:].unsqueeze(0).unsqueeze(0)

                output = diffusion.ddim_sample_loop(
                    model,
                    (local_low_res.shape[0], local_low_res.shape[1], local_low_res.shape[2], local_low_res.shape[3], local_low_res.shape[4]),
                    type=args.type,
                    f_steps=args.f_steps,
                    model_kwargs=local_low_res
                )

                local_low_res= local_low_res.squeeze(0).squeeze(0)
                local_full_res = local_full_res.squeeze(0).squeeze(0)
                output = output.squeeze(0).squeeze(0)

                output_array[i,:,:,:] = output
                low_res_array[i,:,:,:] = local_low_res
                full_res_array[i,:,:,:] = local_full_res

            print(output_array.shape, low_res_array.shape, full_res_array.shape, gpu_id, world_size, actual_patches)
            dist.barrier(device_ids=[gpu_id])

            gathered_output = [th.zeros_like(output_array) for _ in range(world_size)]
            gathered_low_res = [th.zeros_like(low_res_array) for _ in range(world_size)]
            gathered_full_res = [th.zeros_like(full_res_array) for _ in range(world_size)]
            dist.all_gather(gathered_output, output_array)
            dist.all_gather(gathered_low_res, low_res_array)
            dist.all_gather(gathered_full_res, full_res_array)

            assert len(gathered_output) == len(gathered_low_res) == len(gathered_full_res)
            dist.barrier(device_ids=[gpu_id])

            cpu_low_res = [tensor.cpu().detach().numpy() for tensor in gathered_low_res]
            cpu_full_res = [tensor.cpu().detach().numpy() for tensor in gathered_full_res]
            cpu_output = [tensor.cpu().detach().numpy() for tensor in gathered_output]

            if dist.get_rank() == 0:
                low_res_arrays = []
                full_res_arrays = []
                out_arrays = []
                
                for rank in range(world_size):
                    if rank < reminder:
                        actual_patch_count = data_per_gpu + 1
                    else:
                        actual_patch_count = data_per_gpu
                    
                    low_res_arrays.append(cpu_low_res[rank][:actual_patch_count])
                    full_res_arrays.append(cpu_full_res[rank][:actual_patch_count])
                    out_arrays.append(cpu_output[rank][:actual_patch_count])

                temp_low = np.concatenate(low_res_arrays, axis=0)
                temp_full = np.concatenate(full_res_arrays, axis=0)
                temp_out = np.concatenate(out_arrays, axis=0)

                whole_low_res = reconstruct_patches(
                    temp_low, C, H, W,
                    patch_size=tuple(args.patch_size),
                    #stride=tuple(args.patch_size),
                    stride=tuple(s // 1 for s in args.patch_size),
                    mode='overwrite',)
                whole_full_res = reconstruct_patches(
                    temp_full, C, H, W,
                    patch_size=tuple(args.patch_size),
                    #stride=tuple(args.patch_size),
                    stride=tuple(s // 1 for s in args.patch_size),
                    mode='overwrite',)
                whole_out = reconstruct_patches(
                    temp_out, C, H, W,
                    patch_size=tuple(args.patch_size),
                    #stride=tuple(args.patch_size),
                    stride=tuple(s // 1 for s in args.patch_size),
                    mode='overwrite')

                print(whole_low_res.shape, whole_full_res.shape, whole_out.shape, np.max(whole_low_res), np.max(whole_full_res), np.max(whole_out), np.min(whole_low_res), np.min(whole_full_res), np.min(whole_out))

                from skimage.metrics import structural_similarity as ssim
                from pytorch_msssim import ms_ssim
                from metrics import hfen_3d
                PSNR_3D = 0.0
                SSIM_3D = 0.0
                MS_SSIM_3D = 0.0
                HFEN_3D = 0.0
                PSNR_3D = calculate_psnr_3d(whole_out, whole_full_res, data_range=2.0)
                #SSIM_3D = ssim(whole_out, whole_full_res, data_range=2.0, win_size=17, channel_axis=None)
                SSIM_3D = ssim(whole_out, whole_full_res, data_range=2.0, channel_axis=None)
                print('3D PSNR & SSIM calculated.')
                # ssim_2d_list = []
                # for i in range(whole_out.shape[0]):
                #     ssim_val = ssim(
                #         whole_out[i, :, :],
                #         whole_full_res[i, :, :],
                #         data_range=2.0,
                #         channel_axis=None
                #     )
                #     ssim_2d_list.append(ssim_val)
                # SSIM_2D_mean = np.mean(ssim_2d_list)
                # print('2D slice-wise SSIM calculated.')

                out_t  = th.from_numpy(whole_out).unsqueeze(0).unsqueeze(0).float()
                full_t = th.from_numpy(whole_full_res).unsqueeze(0).unsqueeze(0).float()
                MS_SSIM_3D = ms_ssim(
                out_t, full_t,
                data_range=2.0,
                size_average=True).item()
                print('3D MS-SSIM calculated.')
                HFEN_3D = hfen_3d(whole_out, whole_full_res, sigma=1.0)
                print('3D HFEN calculated.')

                # whole_full_res = -whole_full_res
                # whole_low_res = -whole_low_res
                # whole_out = -whole_out

                sitk_low_res = sitk.GetImageFromArray(whole_low_res)
                sitk.WriteImage(sitk_low_res, os.path.join(save_dir, "low_res_volume_{}.nii.gz").format(j))
                sitk_full_res = sitk.GetImageFromArray(whole_full_res)
                sitk.WriteImage(sitk_full_res, os.path.join(save_dir, "full_res_volume_{}.nii.gz").format(j))
                sitk_out = sitk.GetImageFromArray(whole_out)
                sitk.WriteImage(sitk_out, os.path.join(save_dir, "output_volume_{}.nii.gz").format(j))

                # logger.log(
                # "PSNR3D: {}\nSSIM3D: {}\n"
                # .format(
                #     PSNR_3D, SSIM_3D,
                # ))
                
                logger.log(
                "PSNR3D: {}\nSSIM3D: {}\nMS-SSIM3D: {}\nHFEN3D: {}\n"
                .format(
                    PSNR_3D, SSIM_3D, MS_SSIM_3D, HFEN_3D, 
                ))

                # logger.log(
                #     "PSNR3D: {}\nSSIM3D: {}\nSSIM2D_mean: {}\nMS-SSIM3D: {}\nHFEN3D: {}\n"
                #     .format(
                #         PSNR_3D, SSIM_3D, SSIM_2D_mean, MS_SSIM_3D, HFEN_3D, 
                #     ))

                # all_ssim2d.append(SSIM_2D_mean)

                all_psnr.append(PSNR_3D)
                all_ssim.append(SSIM_3D)
                all_msssim.append(MS_SSIM_3D)
                all_hfen.append(HFEN_3D)

            dist.barrier(device_ids=[gpu_id])

        logger.log("sampling complete")
        final_psnr_mean = np.mean(all_psnr);   final_psnr_std = np.std(all_psnr)
        final_ssim_mean = np.mean(all_ssim);   final_ssim_std = np.std(all_ssim)
        final_msssim_mean = np.mean(all_msssim); final_msssim_std = np.std(all_msssim)
        final_hfen_mean = np.mean(all_hfen);   final_hfen_std = np.std(all_hfen)
        
        #final_ssim2d_mean = np.mean(all_ssim2d); final_ssim2d_std = np.std(all_ssim2d)

        # logger.log(
        #     "\n=== Final Results over all samples ===\n"
        #     "PSNR3D: {:.4f} ± {:.4f}\n"
        #     "SSIM3D: {:.4f} ± {:.4f}\n"
        #     "SSIM2D_mean: {:.4f} ± {:.4f}\n"
        #     "MS-SSIM3D: {:.4f} ± {:.4f}\n"
        #     "HFEN3D: {:.4f} ± {:.4f}\n"
        #     .format(
        #         final_psnr_mean, final_psnr_std,
        #         final_ssim_mean, final_ssim_std,
        #         final_ssim2d_mean, final_ssim2d_std,
        #         final_msssim_mean, final_msssim_std,
        #         final_hfen_mean, final_hfen_std,
        #     ))

        logger.log(
            "\n=== Final Results over all samples ===\n"
            "PSNR3D: {:.4f} ± {:.4f}\n"
            "SSIM3D: {:.4f} ± {:.4f}\n"
            "MS-SSIM3D: {:.4f} ± {:.4f}\n"
            "HFEN3D: {:.4f} ± {:.4f}\n"
            .format(
                final_psnr_mean, final_psnr_std,
                final_ssim_mean, final_ssim_std,
                final_msssim_mean, final_msssim_std,
                final_hfen_mean, final_hfen_std,
            ))

        destroy_process_group()



def compute_starts(dim_size, patch, stride):
    starts = list(range(0, dim_size - patch + 1, stride))
    if len(starts) == 0:
        return [0]
    if starts[-1] + patch < dim_size:
        starts.append(dim_size - patch)
    return starts

def reconstruct_patches(patches, C, H, W, patch_size, stride, mode='average'):
    if hasattr(patches, "detach"):
        patches = patches.detach().cpu().numpy()
    patches = np.asarray(patches)

    pd, ph, pw = patch_size
    sd, sh, sw = stride

    c_starts = compute_starts(C, pd, sd)
    h_starts = compute_starts(H, ph, sh)
    w_starts = compute_starts(W, pw, sw)

    expected = len(c_starts) * len(h_starts) * len(w_starts)
    if patches.shape[0] != expected:
        raise ValueError(f"Patch count mismatch: got {patches.shape[0]}, expected {expected} "
                         f"(C,H,W)=({C},{H},{W}), patch={patch_size}, stride={stride}")

    vol = np.zeros((C, H, W), dtype=patches.dtype)

    if mode == 'average':
        weight = np.zeros((C, H, W), dtype=np.float32)
        k = 0
        for c in c_starts:
            for h in h_starts:
                for w in w_starts:
                    p = patches[k]
                    vol[c:c+pd, h:h+ph, w:w+pw] += p
                    weight[c:c+pd, h:h+ph, w:w+pw] += 1.0
                    k += 1
        weight[weight == 0] = 1.0
        vol = vol / weight
        return vol

    elif mode == 'overwrite':
        k = 0
        for c in c_starts:
            for h in h_starts:
                for w in w_starts:
                    vol[c:c+pd, h:h+ph, w:w+pw] = patches[k]
                    k += 1
        return vol

    else:
        raise ValueError("mode must be 'average' or 'overwrite'")


def calculate_psnr_3d(img1, img2, data_range=None):
    if img1.shape != img2.shape:
        raise ValueError("Unmatched shape between images!")

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    diff = (img1 - img2).ravel()
    mse = np.mean(diff ** 2)
    if mse == 0:
        return np.inf

    if data_range is None:
        lo = min(img1.min(), img2.min())
        hi = max(img1.max(), img2.max())
        data_range = hi - lo

    return 10.0 * np.log10((data_range ** 2) / mse)


def load_superres_data_sample(data_dir, batch_size, img_size):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=img_size)

    for pet_batch in data:
        yield pet_batch
                



if __name__ == "__main__":
    main()