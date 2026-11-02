#!/usr/bin/env bash
# Shared-checkpoint patch-only test of learned weighting and auxiliary losses.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="${project_dir}/src"
export HF_HOME="${project_dir}/cache/huggingface"

python="${project_dir}/.venv/bin/python"
root="${project_dir}/runs/patch-only-20261031"
logs="${root}/logs"
previous="${project_dir}/runs/validation-20261031/logs/patch-mechanism-progress.log"
mkdir -p "$logs" "$HF_HOME"

record_exit() {
  status="$?"
  printf 'factorial_exit_code=%s ended_utc=%s\n' "$status" "$(date -u +%FT%TZ)" \
    >> "${logs}/progress.log"
}
trap record_exit EXIT

for attempt in $(seq 1 720); do
  if test -f "$previous" && grep -q '^mechanism_completed=updated ' "$previous"; then
    break
  fi
  if test "$attempt" -eq 720; then
    echo 'Timed out waiting for the legacy-model mechanism audit' >&2
    exit 1
  fi
  sleep 30
done

run_stage() {
  name="$1"
  shift
  printf 'starting=%s at=%s\n' "$name" "$(date -u +%FT%TZ)" | tee -a "${logs}/progress.log"
  "$@" > "${logs}/${name}.log" 2>&1
  printf 'completed=%s at=%s\n' "$name" "$(date -u +%FT%TZ)" | tee -a "${logs}/progress.log"
}

run_stage historical-patch-only-uniform-seed42 \
  "$python" -m fyp_drift.training.classifier \
    --method baseline --pooling-mode patch_only --cv-val-year 2009 \
    --epochs 3 --batch-size 8 --seed 42 --data-root data \
    --output "$root" --run-id historical-patch-only-uniform-seed42 \
    --log-every 100

base="${root}/baseline/historical-patch-only-uniform-seed42/models/best.pt"
test -f "$base"

run_stage updated-patch-only-uniform-seed42 \
  "$python" -m fyp_drift.training.fine_tune \
    --base-checkpoint "$base" --epochs 5 --batch-size 8 \
    --selection-fraction 0.5 --seed 42 --data-root data \
    --output "$root" --run-id updated-patch-only-uniform-seed42 \
    --log-every 50 --drift-scenarios \
    clean illumination white_balance sensor_noise defocus_blur motion_blur \
    jpeg_compression resolution_loss viewpoint_shift occlusion realistic_mix

common=(
  --base-checkpoint "$base" --long-horizon
  --adaptation-profile realistic --initialize-saliency-from-uniform
  --epochs 5 --batch-size 8 --selection-fraction 0.5
  --seed 42 --data-root data --output "$root" --log-every 50
  --evaluate-drift --drift-scenarios
  clean illumination white_balance sensor_noise defocus_blur motion_blur
  jpeg_compression resolution_loss viewpoint_shift occlusion realistic_mix
)

run_stage updated-patch-only-saliency-noaux-seed42 \
  "$python" -m fyp_drift.training.temporal_router \
    "${common[@]}" --disable-saliency-loss \
    --run-id updated-patch-only-saliency-noaux-seed42

run_stage updated-patch-only-saliency-aux-seed42 \
  "$python" -m fyp_drift.training.temporal_router \
    "${common[@]}" \
    --run-id updated-patch-only-saliency-aux-seed42

saliency_run="${root}/siamese/updated-patch-only-saliency-aux-seed42"
run_stage patch-only-router-feature-ablation \
  "$python" -m fyp_drift.evaluation.router_feature_ablation \
    --run-dir "$saliency_run" --data-root data \
    --output "${root}/feature-ablation"

run_stage patch-only-updated-mechanism \
  "$python" -m fyp_drift.evaluation.patch_mechanism \
    --run-dir "$saliency_run" --data-root data --year 2014 \
    --batch-size 16 --role updated \
    --output "${root}/patch-mechanism-updated-2014.json"
