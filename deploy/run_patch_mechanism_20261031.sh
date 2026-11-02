#!/usr/bin/env bash
# Assess the patch-weight mechanism after the corrected conventional run releases the GPU.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="${project_dir}/src"
export HF_HOME="${project_dir}/cache/huggingface"

python="${project_dir}/.venv/bin/python"
root="${project_dir}/runs/validation-20261031"
logs="${root}/logs"
progress="${logs}/finetune-progress.log"
source_run="${root}/siamese/realistic-saliency-seed42"
mkdir -p "$logs"

while ! grep -q '^finetune_exit_code=' "$progress"; do
  sleep 30
done
printf 'mechanism_started_utc=%s\n' "$(date -u +%FT%TZ)" >> "${logs}/patch-mechanism-progress.log"

for role in historical updated; do
  "$python" -m fyp_drift.evaluation.patch_mechanism \
    --run-dir "$source_run" --data-root data --year 2014 \
    --batch-size 16 --role "$role" \
    --output "${root}/patch-mechanism-${role}-2014.json" \
    > "${logs}/patch-mechanism-${role}-2014.log" 2>&1
  printf 'mechanism_completed=%s at=%s\n' "$role" "$(date -u +%FT%TZ)" \
    >> "${logs}/patch-mechanism-progress.log"
done
