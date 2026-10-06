#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_dir="${1:-runs/siamese/20261005-drift-v2-r2-realistic-saliency}"
base_checkpoint="${2:-/mnt/sdd/FYP-fork/runs/xai/20260928T161159Z-horizon-xai/models/best.pt}"
output_root="${3:-reports/transparent_router/20261006-drift-v2}"

mkdir -p "$output_root"
for scenario in motion_blur viewpoint_shift realistic_mix; do
  echo "START ${scenario}"
  tools/uv run --no-sync audit_counterexamples.py \
    --base-checkpoint "$base_checkpoint" \
    --run-dir "$run_dir" \
    --output "$output_root/audit-${scenario}" \
    --year 2014 \
    --scenario "$scenario" \
    --severity 3 \
    --batch-size 16 \
    --top-k 2
  echo "DONE ${scenario}"
done
