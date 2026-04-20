#!/bin/bash

module load conda
conda activate /blue/weishao/share/hxj/conda/envs/3d

echo "Using TMPDIR=$TMPDIR"

torchrun --standalone --nproc_per_node=2 super_res_train.py --config="configs/ps128_ct.yml" --resume_checkpoint=""