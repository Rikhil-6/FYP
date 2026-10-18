#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
storage_dir="$(dirname -- "$(readlink -f -- "${project_dir}/runs")")"
cache_dir="$(readlink -f -- "${project_dir}/cache")"
action="${1:-status}"
if [[ "$#" -gt 0 ]]; then shift; fi

usage() {
  echo "Usage: bash deploy/server.sh {setup|status|smoke|ingest|audit-data|experiment|baseline|xai|siamese|evaluate|drift|all} [model options]"
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
export UV_CACHE_DIR="${cache_dir}/uv"
export UV_PROJECT_ENVIRONMENT="${storage_dir}/venv"
export HF_HOME="${cache_dir}/huggingface"
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
    if ! "$uv" run --no-sync fyp-hardware --require-gpu; then
      echo "Full training requires an accessible GPU by default. Set FYP_ALLOW_CPU_TRAIN=1 to run on CPU." >&2
      exit 2
    fi
  fi
}

train_method() {
  local method="$1"
  shift
  "$uv" run --no-sync fyp-train --method "$method" "$@"
}

train_twin() {
  local base
  base="$(latest_best xai)"
  "$uv" run --no-sync fyp-train-router --base-checkpoint "$base" "$@"
}

case "$action" in
  status)
    git status --short --branch
    "$uv" run --no-sync fyp-hardware
    for name in data runs tools .venv cache; do
      printf '%s -> %s\n' "$name" "$(readlink -f "$name")"
    done
    ;;
  smoke) "$uv" run --no-sync fyp-smoke "$@" ;;
  ingest) "$uv" run --no-sync fyp-ingest "$@" ;;
  audit-data) "$uv" run --no-sync fyp-audit-data "$@" ;;
  experiment) check_training_device; "$uv" run --no-sync fyp-experiment "$@" ;;
  baseline) check_training_device; train_method baseline "$@" ;;
  xai) check_training_device; train_method xai "$@" ;;
  siamese) check_training_device; train_twin "$@" ;;
  evaluate)
    for method in baseline xai; do
      checkpoint="$(latest_best "$method")"
      "$uv" run --no-sync fyp-evaluate --method "$method" --checkpoint "$checkpoint" --output "$(dirname "$(dirname "$checkpoint")")/metrics/recheck" "$@"
    done
    ;;
  drift)
    for method in baseline xai; do
      checkpoint="$(latest_best "$method")"
      run_root="$(dirname "$(dirname "$checkpoint")")"
      "$uv" run --no-sync fyp-evaluate-drift --method "$method" --checkpoint "$checkpoint" \
        --output "${run_root}/metrics/drift" "$@"
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
