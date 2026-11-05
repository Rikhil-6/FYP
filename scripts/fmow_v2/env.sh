# Environment for the fMoW v2 experiments (source before running any fmow_v2 module).
# Set FYP_ROOT to a folder with room for the data (~40 GB), caches and runs; every path below derives from it and can
# also be overridden one by one before sourcing, e.g.  FYP_ROOT=$HOME/fyp source scripts/fmow_v2/env.sh
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
export FYP_ROOT="${FYP_ROOT:-/mnt/sdd/FYP}"
export FYP_DATA="${FYP_DATA:-$FYP_ROOT/data/fmow_v1.1}"
export HF_HOME="${HF_HOME:-$FYP_ROOT/cache/huggingface}"
export FYP_MANIFEST="${FYP_MANIFEST:-$FYP_DATA/subset_14k_country_matched/manifest_downloaded.csv}"
export FYP_METADATA="${FYP_METADATA:-$FYP_DATA/subset_14k_country_matched/acquisition_metadata.csv}"
export FYP_CACHE="${FYP_CACHE:-$FYP_ROOT/cache/fmow42k_u8_280_ctx2.npy}"
export FYP_IMG=280 FYP_CTX=2 FYP_GRID=10
export FYP_RUNS="${FYP_RUNS:-$FYP_ROOT/runs/v2}"
export FYP_TOKENS="${FYP_TOKENS:-$FYP_ROOT/cache/adapter_tokens_v2p}"
export FYP_TOKENS_EXTRA="${FYP_TOKENS_EXTRA:-$FYP_ROOT/cache/adapter_tokens_v2p_extra}"
# Python with the project installed (pip install -e .); falls back to the python on PATH
if [ -z "${PY:-}" ]; then
  if [ -x "$FYP_ROOT/venv/bin/python" ]; then export PY="$FYP_ROOT/venv/bin/python"; else export PY="$(command -v python)"; fi
fi

