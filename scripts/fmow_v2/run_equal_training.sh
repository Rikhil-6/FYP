#!/bin/bash
# Equal-training LoRA and the LoRA + soft-gate hybrid for seeds 42-44 (about 30 min per run on a T4).
set -u
source "$(dirname "$0")/env.sh"
M=fyp_drift.evaluation.fmow_v2
for seed in 42 43 44; do
  $PY -m $M.equal_lora --rank 2 --blocks 6 --condition clean --seed $seed
  $PY -m $M.equal_lora --rank 8 --blocks 6 --condition clean --seed $seed
  $PY -m $M.equal_lora --rank 1 --blocks 2 --condition clean --seed $seed
  $PY -m $M.hybrid --seed $seed
  $PY -m $M.equal_lora --rank 2 --blocks 6 --condition cloud --seed $seed
  $PY -m $M.equal_lora --rank 8 --blocks 6 --condition cloud --seed $seed
done
