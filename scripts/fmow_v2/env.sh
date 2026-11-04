# Environment for the fMoW v2 experiments (source before running any fmow_v2 module).
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
export HF_HOME=/mnt/sdd/FYP/cache/huggingface
export FYP_MANIFEST=/mnt/sdd/FYP/data/fmow_v1.1/subset_14k_country_matched/manifest_downloaded.csv
export FYP_METADATA=/mnt/sdd/FYP/data/fmow_v1.1/subset_14k_country_matched/acquisition_metadata.csv
export FYP_CACHE=/mnt/sdd/FYP/cache/fmow42k_u8_280_ctx2.npy
export FYP_IMG=280 FYP_CTX=2 FYP_GRID=10
export FYP_RUNS=/mnt/sdd/FYP/runs/v2
export FYP_TOKENS=/mnt/sdd/FYP/cache/adapter_tokens_v2p
export FYP_TOKENS_EXTRA=/mnt/sdd/FYP/cache/adapter_tokens_v2p_extra
export PY=/mnt/sdd/FYP/venv/bin/python
