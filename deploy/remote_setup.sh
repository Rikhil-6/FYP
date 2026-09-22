#!/usr/bin/env bash
set -euo pipefail

project_dir="${HOME}/FYP"
storage_dir="/mnt/sdd/FYP"

if [[ ! -f "${project_dir}/pyproject.toml" || ! -f "${project_dir}/uv.lock" ]]; then
  echo "Expected the FYP checkout at ${project_dir}" >&2
  exit 1
fi
if [[ ! -w /mnt/sdd ]]; then
  echo "The mounted data volume /mnt/sdd is not writable" >&2
  exit 1
fi

mkdir -p "${storage_dir}/data" "${storage_dir}/runs" "${storage_dir}/cache/uv" "${storage_dir}/tools"

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

link_dir data "${storage_dir}/data"
link_dir runs "${storage_dir}/runs"
link_dir tools "${storage_dir}/tools"

if [[ ! -x "${storage_dir}/tools/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${storage_dir}/tools" sh
fi

export UV_CACHE_DIR="${storage_dir}/cache/uv"
export UV_PROJECT_ENVIRONMENT="${storage_dir}/venv"
"${storage_dir}/tools/uv" sync --locked --project "$project_dir"
link_dir .venv "${storage_dir}/venv"

echo "FYP ready: ${project_dir}"
echo "Persistent data: ${storage_dir}/data"
echo "Persistent runs: ${storage_dir}/runs"
