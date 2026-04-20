#!/bin/bash

module load conda
conda activate hxj_b200
export OMP_NUM_THREADS=8

echo "Using TMPDIR=$TMPDIR"

torchrun --standalone --nproc_per_node=1 super_res_train.py --config="configs/ps128_pet.yml" --resume_checkpoint=""