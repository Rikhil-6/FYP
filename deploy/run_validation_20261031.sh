#!/usr/bin/env bash
# Corrected CLEAR validation on the isolated server checkout.
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
  printf 'exit_code=%s ended_utc=%s\n' "$status" "$(date -u +%FT%TZ)" >> "${logs}/progress.log"
}
trap record_exit EXIT

run_stage() {
  name="$1"
  shift
  printf 'starting=%s at=%s\n' "$name" "$(date -u +%FT%TZ)" | tee -a "${logs}/progress.log"
  "$@" > "${logs}/${name}.log" 2>&1
  printf 'completed=%s at=%s\n' "$name" "$(date -u +%FT%TZ)" | tee -a "${logs}/progress.log"
}

"$python" -m fyp_drift.system.hardware --require-gpu > "${logs}/hardware.json"
printf 'source_commit=73cd4e1\nhelper_sha256=%s\n' \
  "$(sha256sum src/fyp_drift/evaluation/router_feature_ablation.py | cut -d ' ' -f 1)" \
  > "${logs}/code_provenance.txt"

base="${root}/xai/historical-xai-2009-seed42/models/best.pt"
run_stage historical-xai-2009-seed42 \
  "$python" -m fyp_drift.training.classifier \
    --method xai --cv-val-year 2009 --epochs 3 --batch-size 8 \
    --seed 42 --data-root data --output "$root" \
    --run-id historical-xai-2009-seed42 --log-every 100
test -f "$base"

run_stage realistic-saliency-seed42 \
  "$python" -m fyp_drift.training.temporal_router \
    --base-checkpoint "$base" --long-horizon \
    --adaptation-profile realistic --epochs 5 --batch-size 8 \
    --seed 42 --data-root data --output "$root" \
    --run-id realistic-saliency-seed42 --log-every 50 --evaluate-drift

run_stage realistic-noaux-seed42 \
  "$python" -m fyp_drift.training.temporal_router \
    --base-checkpoint "$base" --long-horizon \
    --adaptation-profile realistic --disable-saliency-loss \
    --epochs 5 --batch-size 8 --seed 42 --data-root data \
    --output "$root" --run-id realistic-noaux-seed42 \
    --log-every 50 --evaluate-drift

run_stage feature-ablation-seed42 \
  "$python" -m fyp_drift.evaluation.router_feature_ablation \
    --run-dir "${root}/siamese/realistic-saliency-seed42" \
    --data-root data --output "${root}/feature-ablations/seed42"
