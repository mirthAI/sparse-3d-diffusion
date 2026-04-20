module load conda
conda activate hxj_b200

export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node=2 super_res_sample_3d.py --config="configs/ps128_pet.yml" --model_path="/red/weishao/weishao/DPM_3D/output_PET/ckpt_fast_ts5_x0_ch64_s222_pet/ema_0.99_200000.pt"