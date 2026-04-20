#!/usr/bin/env bash
set -euo pipefail

# ===== 环境 =====
module load conda
conda activate hxj_b200
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1   # 让日志实时flush

# ===== 常量 =====
CONFIG="configs/ps128_pet.yml"
CKPT_DIR="/red/weishao/weishao/DPM_3D/output_PET/ckpt_fast_ts5_x0_ch64_s222_pet"
SCRIPT="super_res_sample_3d.py"
NPROC=2

# 日志目录
LOG_DIR="logs/fast_pet"
mkdir -p "$LOG_DIR"

# 如果你的脚本会把结果写到某个输出目录，也可以根据 step 建子目录
# 例如：OUTPUT_ROOT="results/super_res_sample_3d"; mkdir -p "$OUTPUT_ROOT"

# ===== 要跑的 step 列表：40000, 60000, ..., 200000 =====
# 方式1：显式列出（最稳）
STEPS=(20000 40000 60000 80000 100000 120000 140000 160000 180000 200000)

# 方式2：用 brace expansion（两者二选一）
# readarray -t STEPS < <(seq 40000 20000 200000)

# ===== 逐个执行 =====
for s in "${STEPS[@]}"; do
  step=$(printf "%06d" "$s")  # 060000 之类，匹配 model 文件命名
  ckpt="${CKPT_DIR}/ema_0.99_${step}.pt"
  ts=$(date +"%Y%m%d_%H%M%S")
  logf="${LOG_DIR}/step_${step}_${ts}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[WARN] checkpoint 不存在：$ckpt" | tee -a "$logf"
    continue
  fi

  echo "===== RUN step ${step} @ ${ts} =====" | tee -a "$logf"
  echo "ckpt: $ckpt" | tee -a "$logf"

  # 如果你的评估脚本支持指定输出目录，可以加上 --out_dir "results/.../step_${step}"
  # outdir="${OUTPUT_ROOT}/step_${step}"; mkdir -p "$outdir"

  # 正式运行（顺序执行）
  time torchrun --standalone --nproc_per_node="${NPROC}" "$SCRIPT" \
    --config="$CONFIG" \
    --model_path="$ckpt" \
    2>&1 | tee -a "$logf"

  echo "===== DONE step ${step} =====" | tee -a "$logf"
  echo
done

echo "全部完成。日志在：$LOG_DIR"
