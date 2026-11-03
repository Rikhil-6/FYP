set -x
cd ~/FYP-saliency-layer-full-bundle-20261103
export PYTHONPATH=src HF_HOME=/mnt/sdd/FYP/cache/huggingface
PY=/mnt/sdd/FYP/venv/bin/python
OUT=/mnt/sdd/FYP/runs/today/main
$PY fmow_today.py --output $OUT --stages cache features gate calib || exit 1
mkdir -p /mnt/sdd/FYP/runs/today/smoke
for f in features.npz gate.json head.pt calib.json; do ln -sf $OUT/$f /mnt/sdd/FYP/runs/today/smoke/$f; done
$PY fmow_today.py --output /mnt/sdd/FYP/runs/today/smoke --stages arms summary --limit 48 --epochs 1 --seeds 42 --lambdas 1.0 || exit 1
$PY fmow_today.py --output $OUT --stages arms summary
