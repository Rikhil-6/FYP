import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Dinov2Model

from fyp_drift.evaluation.fmow_saliency_layer_pilot import divide_groups, evaluate, loader
from fyp_drift.evaluation.fmow_temporal_pilot import FMoWImages, load_manifest
from fyp_drift.evaluation.fmow_layer_selection import add_group_lora
from fyp_drift.models.vision_transformer import BACKBONE

run = Path('/mnt/sdd/FYP/runs/fmow-saliency-layer-full-seed42-20261103')
out = Path('/mnt/sdd/FYP/runs/fmow-supervised-lambda0-shuffled-check-20261103.json')
if out.exists():
    print(out.read_text())
    raise SystemExit
torch.manual_seed(42)
config = json.loads((run / 'config.json').read_text())
rows, categories = load_manifest(Path(config['manifest']))
adapt_rows = sorted((r for r in rows if r['period'] == 'adapt_2014_15'), key=lambda r: r['local_path'])
future = sorted((r for r in rows if r['period'] == 'future_2016_17'), key=lambda r: r['local_path'])
train, val = divide_groups(adapt_rows, 0.1, 42)
_, test = divide_groups(future, 0.5, 43)
head = nn.Linear(384, len(categories)).cuda().eval()
head.load_state_dict(torch.load(run / 'head.pt', map_location='cuda', weights_only=True))
for p in head.parameters():
    p.requires_grad = False
model = Dinov2Model.from_pretrained(BACKBONE).cuda()
count = add_group_lora(model, 'E')
model.cuda()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')
history = []
start = time.time()
for epoch in range(2):
    model.train()
    losses, entropies, marginal_entropies = [], [], []
    train_data = DataLoader(FMoWImages(train), batch_size=8, shuffle=True,
                            generator=torch.Generator().manual_seed(42 + epoch),
                            num_workers=2, pin_memory=True)
    for images, labels, _ in tqdm(train_data, desc=f'supervised epoch {epoch+1}'):
        with torch.autocast('cuda'):
            tokens = model(pixel_values=images.cuda()).last_hidden_state[:, 1:]
            logits = head(tokens.mean(1).float())
            loss = nn.functional.cross_entropy(logits, labels.cuda())
            prob = logits.softmax(1)
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(1).mean()
            mean_prob = prob.mean(0)
            marginal = -(mean_prob * mean_prob.clamp_min(1e-8).log()).sum()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        entropies.append(entropy.item())
        marginal_entropies.append(marginal.item())
    metrics, _ = evaluate(model, head, loader(val, 8, 2, True), 'cuda')
    point = {'epoch': epoch+1, 'cross_entropy': float(np.mean(losses)),
             'entropy': float(np.mean(entropies)),
             'marginal_entropy': float(np.mean(marginal_entropies)),
             'adapt_val_macro_f1': metrics['macro_f1']}
    history.append(point)
    print(json.dumps(point), flush=True)
metrics, predictions = evaluate(model, head, loader(test, 8, 2, True), 'cuda')
counts = np.bincount(predictions, minlength=len(categories))
result = {'seed': 42, 'group': 'E', 'lambda': 0, 'trainable_parameters': count,
          'train_count': len(train), 'val_count': len(val), 'test_count': len(test),
          'history': history, 'future_test': {k:v for k,v in metrics.items() if k!='per_class_f1'},
          'unique_predictions': int((counts>0).sum()),
          'top_prediction_classes': np.argsort(-counts)[:10].tolist(),
          'top_prediction_counts': counts[np.argsort(-counts)[:10]].tolist(),
          'seconds': time.time()-start}
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
