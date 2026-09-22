#!/usr/bin/env bash
set -euo pipefail

project_dir="${HOME}/FYP"
storage_dir="/mnt/sdd/FYP"
action="${1:-status}"
if [[ "$#" -gt 0 ]]; then shift; fi

usage() {
  echo "Usage: bash deploy/server.sh {setup|status|smoke|ingest|baseline|xai|siamese|evaluate|all} [model options]"
  echo "Set FYP_ALLOW_CPU_TRAIN=1 to permit full training without a GPU."
}

if [[ "$action" == "setup" ]]; then
  exec bash "${project_dir}/deploy/remote_setup.sh"
fi
if [[ ! -x "${project_dir}/tools/uv" || ! -d "${storage_dir}/venv" ]]; then
  echo "Run 'bash deploy/server.sh setup' first" >&2
  exit 1
fi

cd "$project_dir"
export UV_CACHE_DIR="${storage_dir}/cache/uv"
export UV_PROJECT_ENVIRONMENT="${storage_dir}/venv"
export HF_HOME="${storage_dir}/cache/huggingface"
mkdir -p "$HF_HOME"
uv="${project_dir}/tools/uv"

latest_best() {
  local method="$1"
  local checkpoint
  checkpoint="$(find "${project_dir}/runs/${method}" -mindepth 3 -maxdepth 3 -path '*/models/best.pt' -type f 2>/dev/null | sort | tail -n 1)"
  if [[ -z "$checkpoint" ]]; then
    echo "No best checkpoint found for ${method}" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint"
}

check_training_device() {
  if [[ "${FYP_ALLOW_CPU_TRAIN:-0}" != "1" ]]; then
    if ! "$uv" run --no-sync hardware.py --require-gpu; then
      echo "Full training requires an accessible GPU by default. Set FYP_ALLOW_CPU_TRAIN=1 to run on CPU." >&2
      exit 2
    fi
  fi
}

train_method() {
  local method="$1"
  shift
  "$uv" run --no-sync train.py --method "$method" "$@"
}

train_twin() {
  local base
  base="$(latest_best xai)"
  "$uv" run --no-sync train_siamese.py --base-checkpoint "$base" "$@"
}

case "$action" in
  status)
    git status --short --branch
    "$uv" run --no-sync hardware.py
    for name in data runs tools .venv cache; do
      printf '%s -> %s\n' "$name" "$(readlink -f "$name")"
    done
    ;;
  smoke) "$uv" run --no-sync smoke.py "$@" ;;
  ingest) "$uv" run --no-sync ingest.py "$@" ;;
  baseline) check_training_device; train_method baseline "$@" ;;
  xai) check_training_device; train_method xai "$@" ;;
  siamese) check_training_device; train_twin "$@" ;;
  evaluate)
    for method in baseline xai; do
      checkpoint="$(latest_best "$method")"
      "$uv" run --no-sync evaluate.py --method "$method" --checkpoint "$checkpoint" --output "$(dirname "$(dirname "$checkpoint")")/metrics/recheck" "$@"
    done
    ;;
  all)
    check_training_device
    train_method baseline "$@"
    train_method xai "$@"
    train_twin "$@"
    ;;
  *) usage >&2; exit 2 ;;
esac
