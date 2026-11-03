while tmux has-session -t today 2>/dev/null; do sleep 60; done
set -x
cd ~/FYP-saliency-layer-full-bundle-20261103
export PYTHONPATH=src HF_HOME=/mnt/sdd/FYP/cache/huggingface
/mnt/sdd/FYP/venv/bin/python -m fyp_drift.evaluation.fmow_dinov2l_study --output /mnt/sdd/FYP/runs/today/main --stages unseen_cloud
