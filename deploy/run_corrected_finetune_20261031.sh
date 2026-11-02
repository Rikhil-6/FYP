#!/usr/bin/env bash
# Complete the corrected CLEAR comparison with a conventional uniform-pooling arm.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="${project_dir}/src"
export HF_HOME="${project_dir}/cache/huggingface"

python="${project_dir}/.venv/bin/python"
root="${project_dir}/runs/validation-20261031"
logs="${root}/logs"
mkdir -p "$logs" "$HF_HOME"

record_exit() {
  status="$?"
  printf 'finetune_exit_code=%s ended_utc=%s\n' "$status" "$(date -u +%FT%TZ)" >> "${logs}/finetune-progress.log"
}
trap record_exit EXIT

run_stage() {
  name="$1"
  shift
  printf 'starting=%s at=%s\n' "$name" "$(date -u +%FT%TZ)" | tee -a "${logs}/finetune-progress.log"
  "$@" > "${logs}/${name}.log" 2>&1
  printf 'completed=%s at=%s\n' "$name" "$(date -u +%FT%TZ)" | tee -a "${logs}/finetune-progress.log"
}

base="${root}/baseline/historical-uniform-2009-seed42/models/best.pt"
run_stage historical-uniform-2009-seed42 \
  "$python" -m fyp_drift.training.classifier \
    --method baseline --cv-val-year 2009 --epochs 3 --batch-size 8 \
    --seed 42 --data-root data --output "$root" \
    --run-id historical-uniform-2009-seed42 --log-every 100
test -f "$base"

run_stage corrected-finetune-seed42 \
  "$python" -m fyp_drift.training.fine_tune \
    --base-checkpoint "$base" --epochs 5 --batch-size 8 \
    --selection-fraction 0.5 --seed 42 --data-root data \
    --output "$root" --run-id corrected-finetune-seed42 \
    --log-every 50 --drift-scenarios \
    clean illumination white_balance sensor_noise defocus_blur motion_blur \
    jpeg_compression resolution_loss viewpoint_shift occlusion realistic_mix
