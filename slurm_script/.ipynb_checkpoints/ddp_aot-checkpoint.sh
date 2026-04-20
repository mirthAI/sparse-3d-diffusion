#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=196gb
#SBATCH --partition=hpg-b200
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --job-name=fast_ts5_x0_vloss_aot_modulation_encoder
#SBATCH --time=48:00:00
#SBATCH --output=/blue/weishao/hongxu.jiang/3d_diffusion/Output/AOT/%x.%j.out

module load cuda/12.8.1
module load conda
conda activate /blue/weishao/share/hxj/conda/envs/3d

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"

export OMP_NUM_THREADS=8

nodes=($( scontrol show hostnames $SLURM_JOB_NODELIST ))
nodes_array=($nodes)
echo Node list $nodes_array
 
head_node_ip=`hostname --ip-address`

echo HeadNodeIP: $head_node_ip
 
head_node_port=29500
export LOGLEVEL=INFO
pwd; hostname; date
 
srun --export=ALL torchrun \
  --standalone \
  --nproc_per_node=2 \
  super_res_train.py  --config="configs/ps128_aot.yml" --resume_checkpoint=""