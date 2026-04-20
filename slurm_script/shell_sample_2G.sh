#!/usr/bin/env bash
set -euo pipefail

module load cuda/12.8.1
module load conda
conda activate sparse-3d-diffusion
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

CONFIG="configs/ps128_ct.yml"
CKPT_DIR="/red/weishao/weishao/DPM_3D/output_AOT/ckpt_ddpm_x0_ch64_s222_aot"
SCRIPT="super_res_sample_3d.py"
NPROC=2

LOG_DIR="logs/ddim_aot"
mkdir -p "$LOG_DIR"


STEPS=(20000 40000 60000 80000 100000 120000 140000 160000 180000 200000)

for s in "${STEPS[@]}"; do
  step=$(printf "%06d" "$s")
  ckpt="${CKPT_DIR}/model${step}.pt"
  ts=$(date +"%Y%m%d_%H%M%S")
  logf="${LOG_DIR}/step_${step}_${ts}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[WARN] checkpoint not found：$ckpt" | tee -a "$logf"
    continue
  fi

  echo "===== RUN step ${step} @ ${ts} =====" | tee -a "$logf"
  echo "ckpt: $ckpt" | tee -a "$logf"

  time torchrun --standalone --nproc_per_node="${NPROC}" "$SCRIPT" \
    --config="$CONFIG" \
    --model_path="$ckpt" \
    2>&1 | tee -a "$logf"

  echo "===== DONE step ${step} =====" | tee -a "$logf"
  echo
done

echo "Complte, logged into：$LOG_DIR"
