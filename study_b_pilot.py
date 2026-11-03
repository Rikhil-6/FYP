"""One-seed matched supervised anchoring pilot. Future labels are evaluation only."""
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

source = Path('/mnt/sdd/FYP/runs/fmow-saliency-layer-full-seed42-20261103')
output = Path('/mnt/sdd/FYP/runs/fmow-study-b-anchoring-seed42-20261103')
output.mkdir(parents=True, exist_ok=True)
config = json.loads((source / 'config.json').read_text())
rows, categories = load_manifest(Path(config['manifest']))
adapt_rows = sorted((r for r in rows if r['period'] == 'adapt_2014_15'), key=lambda r: r['local_path'])
future = sorted((r for r in rows if r['period'] == 'future_2016_17'), key=lambda r: r['local_path'])
train, val = divide_groups(adapt_rows, 0.1, 42)
_, test = divide_groups(future, 0.5, 43)
torch.manual_seed(42)
head = nn.Linear(384, len(categories)).cuda().eval()
head.load_state_dict(torch.load(source / 'head.pt', map_location='cuda', weights_only=True))
for p in head.parameters():
    p.requires_grad = False
reference = Dinov2Model.from_pretrained(BACKBONE).cuda().eval()
for p in reference.parameters():
    p.requires_grad = False

for mode in ('uniform', 'shuffled', 'saliency'):
    path = output / f'{mode}.json'
    if path.exists():
        print(f'skip {path}', flush=True)
        continue
    torch.manual_seed(42)
    model = Dinov2Model.from_pretrained(BACKBONE).cuda()
    count = add_group_lora(model, 'E')
    model.cuda()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    start = time.time()
    for epoch in range(2):
        model.train()
        train_data = DataLoader(FMoWImages(train), batch_size=8, shuffle=True,
                                generator=torch.Generator().manual_seed(42 + epoch),
                                num_workers=2, pin_memory=True)
        ce_values, anchor_values = [], []
        for batch_id, (images, labels, _) in enumerate(tqdm(train_data, desc=f'{mode} epoch {epoch+1}')):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad(), torch.autocast('cuda'):
                ref = reference(pixel_values=images).last_hidden_state[:, 1:].float()
                pred = head(ref.mean(1)).argmax(1)
                evidence = (ref * head.weight[pred, None, :]).sum(-1)
                top = evidence.topk(round(0.4 * evidence.shape[1]), dim=1).indices
                weights = torch.zeros_like(evidence)
                weights.scatter_(1, top, 1.0 / top.shape[1])
                if mode == 'uniform':
                    weights.fill_(1.0 / evidence.shape[1])
                elif mode == 'shuffled':
                    random_order = torch.rand(evidence.shape,
                                              generator=torch.Generator().manual_seed(42+epoch*10000+batch_id),
                                              device='cpu').argsort(1).to('cuda')
                    weights = weights.gather(1, random_order)
            with torch.autocast('cuda'):
                tokens = model(pixel_values=images).last_hidden_state[:, 1:].float()
                logits = head(tokens.mean(1))
                ce = nn.functional.cross_entropy(logits, labels)
                anchor = (weights * (tokens - ref).square().mean(-1)).sum(1).mean()
                loss = ce + 0.1 * anchor
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ce_values.append(ce.item())
            anchor_values.append(anchor.item())
        val_metrics, _ = evaluate(model, head, loader(val, 8, 2, True), 'cuda')
        point = {'epoch': epoch+1, 'cross_entropy': float(np.mean(ce_values)),
                 'anchor_mse': float(np.mean(anchor_values)),
                 'adapt_val_macro_f1': val_metrics['macro_f1']}
        history.append(point)
        print(json.dumps({'mode': mode, **point}), flush=True)
    test_metrics, predictions = evaluate(model, head, loader(test, 8, 2, True), 'cuda')
    counts = np.bincount(predictions, minlength=len(categories))
    result = {'seed': 42, 'mode': mode, 'lambda': 0.1, 'group': 'E',
              'trainable_parameters': count, 'train_count': len(train),
              'val_count': len(val), 'test_count': len(test), 'history': history,
              'future_test': {k:v for k,v in test_metrics.items() if k!='per_class_f1'},
              'unique_predictions': int((counts>0).sum()), 'seconds': time.time()-start}
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    del model
    torch.cuda.empty_cache()
