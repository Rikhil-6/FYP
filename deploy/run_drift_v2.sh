#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
storage_dir="$(dirname -- "$(readlink -f -- "${project_dir}/runs")")"
uv="${project_dir}/tools/uv"
epochs="${EPOCHS:-5}"
batch_size="${BATCH_SIZE:-8}"
seed="${SEED:-42}"
stamp="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
base_checkpoint="${BASE_CHECKPOINT:-}"

if [[ -z "$base_checkpoint" ]]; then
  base_checkpoint="$(find "${project_dir}/runs/xai" -mindepth 3 -maxdepth 3 \
    -path '*/models/best.pt' -type f | sort | tail -n 1)"
fi
if [[ ! -f "$base_checkpoint" ]]; then
  echo "Historical XAI checkpoint missing: $base_checkpoint" >&2
  exit 1
fi

export UV_CACHE_DIR="$(readlink -f -- "${project_dir}/cache")/uv"
export UV_PROJECT_ENVIRONMENT="${storage_dir}/venv"
export HF_HOME="$(readlink -f -- "${project_dir}/cache")/huggingface"
mkdir -p "$HF_HOME" "${project_dir}/runs/experiments/drift-v2-${stamp}"
log="${project_dir}/runs/experiments/drift-v2-${stamp}/console.log"

cd "$project_dir"
"$uv" run --no-sync fyp-hardware --require-gpu

run_profile() {
  local suffix="$1"
  shift
  local run_dir="${project_dir}/runs/siamese/${stamp}-realistic-${suffix}"
  local -a resume_args=()
  if [[ -f "${run_dir}/config.json" && -f "${run_dir}/models/resume.pt" ]]; then
    echo "Resuming ${suffix} from ${run_dir}/models/resume.pt"
    resume_args=(--resume-run-dir "$run_dir")
  fi
  "$uv" run --no-sync fyp-train-router \
    --base-checkpoint "$base_checkpoint" \
    --run-id "${stamp}-realistic-${suffix}" \
    "${resume_args[@]}" \
    --long-horizon \
    --adaptation-profile realistic \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --seed "$seed" \
    --log-every 50 \
    --evaluate-drift \
    "$@"
}

{
  echo "Drift v2 matched run: ${stamp}"
  echo "Historical checkpoint: ${base_checkpoint}"
  echo "Epochs: ${epochs}; batch size: ${batch_size}; seed: ${seed}"
  echo "Starting saliency-supervised profile"
  run_profile saliency
  echo "Starting matched no-auxiliary-saliency control"
  run_profile noaux --disable-saliency-loss
  echo "Completed drift v2 matched run"
} 2>&1 | tee -a "$log"

echo "Log: $log"
