#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
storage_dir="${FYP_STORAGE_DIR:-/mnt/sdd/FYP}"
data_dir="${FYP_DATA_DIR:-${storage_dir}/data}"
cache_dir="${FYP_CACHE_DIR:-${storage_dir}/cache}"
tools_dir="${FYP_TOOLS_DIR:-${storage_dir}/tools}"

if [[ ! -f "${project_dir}/pyproject.toml" || ! -f "${project_dir}/uv.lock" ]]; then
  echo "Expected the FYP checkout at ${project_dir}" >&2
  exit 1
fi
if [[ ! -w /mnt/sdd ]]; then
  echo "The mounted data volume /mnt/sdd is not writable" >&2
  exit 1
fi

mkdir -p "${data_dir}" "${storage_dir}/runs" "${cache_dir}/uv" "${tools_dir}"

link_dir() {
  local name="$1"
  local destination="$2"
  local link="${project_dir}/${name}"
  if [[ -L "$link" ]]; then
    if [[ "$(readlink -f "$link")" != "$(readlink -f "$destination")" ]]; then
      echo "Existing ${link} points elsewhere" >&2
      exit 1
    fi
  elif [[ -e "$link" ]]; then
    echo "Existing ${link} is not a symlink; refusing to replace it" >&2
    exit 1
  else
    ln -s "$destination" "$link"
  fi
}

link_dir data "${data_dir}"
link_dir runs "${storage_dir}/runs"
link_dir tools "${tools_dir}"
link_dir cache "${cache_dir}"

if [[ ! -x "${tools_dir}/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${tools_dir}" sh
fi

export UV_CACHE_DIR="${cache_dir}/uv"
export UV_PROJECT_ENVIRONMENT="${storage_dir}/venv"
"${tools_dir}/uv" sync --locked --project "$project_dir"
link_dir .venv "${storage_dir}/venv"

echo "FYP ready: ${project_dir}"
echo "Persistent data: ${data_dir}"
echo "Persistent runs: ${storage_dir}/runs"
