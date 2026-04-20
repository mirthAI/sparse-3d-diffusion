#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --partition=hpg-b200
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --job-name=fast_ts5_x0_ch48_s222_ct
#SBATCH --time=96:00:00
#SBATCH --output=/red/weishao/weishao/DPM_3D/output_CT/%x.%j.out

module load cuda/12.8.1
module load conda
conda activate /blue/weishao/share/hxj/conda/envs/hxj_fast

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
  super_res_train.py  --config="configs/ps128_ct.yml" --resume_checkpoint=""