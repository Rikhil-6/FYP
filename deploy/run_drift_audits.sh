#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <router-run-dir> <historical-checkpoint> [output-dir]" >&2
  exit 2
fi

run_dir="$1"
base_checkpoint="$2"
output_root="${3:-runs/audits/drift}"

mkdir -p "$output_root"
for scenario in motion_blur viewpoint_shift realistic_mix; do
  echo "START ${scenario}"
  tools/uv run --no-sync fyp-audit-router \
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
