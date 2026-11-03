set -x
cd ~/FYP-saliency-layer-full-bundle-20261103
export PYTHONPATH=src HF_HOME=/mnt/sdd/FYP/cache/huggingface
PY=/mnt/sdd/FYP/venv/bin/python
M=fyp_drift.evaluation.fmow_dinov2l_study
rm -rf /mnt/sdd/FYP/runs/today/smoke/arms
$PY -m $M --output /mnt/sdd/FYP/runs/today/smoke --stages arms summary --limit 48 --epochs 1 --seeds 42 --lambdas 1.0 || exit 1
$PY -m $M --output /mnt/sdd/FYP/runs/today/main --stages arms summary
