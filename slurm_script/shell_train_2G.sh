#!/bin/bash

module load conda
conda sparse-3d-diffusion

echo "Using TMPDIR=$TMPDIR"

torchrun --standalone --nproc_per_node=2 super_res_train.py --config="configs/ps128_ct.yml" --resume_checkpoint=""