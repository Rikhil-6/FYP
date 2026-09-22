#!/usr/bin/env bash
# Run ingestion and training in order while retaining separate stage logs.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ "${1:-}" == "--detach" ]]; then
  shift
  if ! command -v tmux >/dev/null; then
    echo "tmux is required for --detach" >&2
    exit 2
  fi
  session="fyp-$(date -u +%Y%m%dT%H%M%SZ)"
  printf -v command '%q ' bash "${project_dir}/deploy/launch.sh" "$@"
  tmux new-session -d -s "$session" "$command"
  echo "Started tmux session: $session"
  echo "Watch: tmux attach -t $session"
  echo "Logs: ${project_dir}/runs/pipelines/"
  exit 0
fi

mode="${1:-all}"
if [[ "$#" -gt 0 ]]; then shift; fi
if [[ "$mode" != "all" && "$mode" != "ingest" && "$mode" != "train" ]]; then
  echo "Usage: bash deploy/launch.sh [--detach] {all|ingest|train} [experiment options]" >&2
  exit 2
fi
if [[ "$mode" == "ingest" && "$#" -gt 0 ]]; then
  echo "Experiment options apply to all or train mode" >&2
  exit 2
fi

run_dir="${project_dir}/runs/pipelines/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"
data_dir="$(readlink -f -- "${project_dir}/data")"
if [[ ! -d "$data_dir" || ! -w "$data_dir" ]]; then
  echo "Data directory is missing or not writable: $data_dir" >&2
  exit 2
fi
record_exit() {
  local status="$?"
  printf 'exit_code=%s\nended_utc=%s\n' "$status" "$(date -u +%FT%TZ)" >> "${run_dir}/run.txt"
}
trap record_exit EXIT
printf 'mode=%s\ncommit=%s\nstarted_utc=%s\n' \
  "$mode" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "${run_dir}/run.txt"
echo "Pipeline logs: $run_dir"
echo "Data directory: ${project_dir}/data -> $data_dir"

if [[ "$mode" == "all" || "$mode" == "ingest" ]]; then
  echo "Starting FMoW ingestion"
  bash deploy/server.sh ingest 2>&1 | tee "${run_dir}/ingest.log"
fi
if [[ "$mode" == "all" || "$mode" == "train" ]]; then
  echo "Starting GPU experiment"
  bash deploy/server.sh experiment "$@" 2>&1 | tee "${run_dir}/train.log"
fi
echo "Pipeline completed: $run_dir"
