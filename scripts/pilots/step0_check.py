import json
from pathlib import Path

import torch
from torch import nn
from transformers import Dinov2Model

from fyp_drift.evaluation.fmow_saliency_layer_pilot import divide_groups, loader
from fyp_drift.evaluation.fmow_temporal_pilot import load_manifest
from fyp_drift.evaluation.fmow_layer_selection import add_group_lora
from fyp_drift.models.vision_transformer import BACKBONE

run = Path('/mnt/sdd/FYP/runs/fmow-saliency-layer-full-seed42-20261103')
config = json.loads((run / 'config.json').read_text())
rows, categories = load_manifest(Path(config['manifest']))
future = sorted((row for row in rows if row['period'] == 'future_2016_17'), key=lambda row: row['local_path'])
_, test = divide_groups(future, 0.5, 43)
images, labels, _ = next(iter(loader(test[:8], 8, 0, True)))
images = images.cuda()
head = nn.Linear(384, len(categories)).cuda().eval()
head.load_state_dict(torch.load(run / 'head.pt', map_location='cuda', weights_only=True))
model = Dinov2Model.from_pretrained(BACKBONE).cuda().eval()
with torch.inference_mode(), torch.autocast('cuda'):
    before = head(model(pixel_values=images).last_hidden_state[:, 1:].mean(1).float())
for group in ('E', 'M', 'L', 'ALL'):
    wrapped = Dinov2Model.from_pretrained(BACKBONE).cuda().eval()
    add_group_lora(wrapped, group)
    wrapped.cuda().eval()
    with torch.inference_mode(), torch.autocast('cuda'):
        after = head(wrapped(pixel_values=images).last_hidden_state[:, 1:].mean(1).float())
    print(json.dumps({'group': group, 'max_logit_diff': (before-after).abs().max().item(),
                      'prediction_match': (before.argmax(1)==after.argmax(1)).sum().item(),
                      'images': len(images)}), flush=True)
