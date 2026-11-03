"""One-day FMoW study on DINOv2-L: harm gate, cloud calibration, matched anchoring arms.

Stages (each resumable; outputs under --output):
  cache    decode every manifest image once to a uint8 224x224 memmap
  features frozen DINOv2-L mean-patch + CLS features (fp16) for every image
  gate     linear head on historical train; scores by window; MMD / domain AUC
  calib    cloud alpha chosen on historical-val so the frozen drop is 5-10 points
  arms     LoRA (last K blocks) supervised adaptation: plain, uniform/shuffled/saliency anchors
           x lambda grid (picked on adapt-val) x condition (clean, cloud) x seeds
  summary  per-arm table + paired clustered bootstrap deltas

Future-test labels are used only for final scoring, never for any choice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
PERIODS = ("historical_2010_13", "adapt_2014_15", "future_2016_17")


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    tmp.replace(path)


# ----------------------------------------------------------------------------- data
def load_rows(manifest):
    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [r for r in rows if r.get("local_path") and Path(r["local_path"]).exists()]
    classes = sorted({r["category"] for r in rows})
    index = {c: i for i, c in enumerate(classes)}
    for i, r in enumerate(rows):
        r["class_id"] = index[r["category"]]
        r["idx"] = i
        r["group"] = f"{r['country']}:{r['utm']}"
    return rows, classes


def divide_groups(rows, fraction, seed):
    groups = [r["group"] for r in rows]
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                .split(rows, groups=groups))
    left, right = [rows[i] for i in a], [rows[i] for i in b]
    assert not ({r["group"] for r in left} & {r["group"] for r in right})
    return left, right


def make_splits(rows):
    by = {p: sorted((r for r in rows if r["period"] == p), key=lambda r: r["local_path"])
          for p in PERIODS}
    hist_train, hist_val = divide_groups(by[PERIODS[0]], 0.15, 42)
    adapt_train, adapt_val = divide_groups(by[PERIODS[1]], 0.1, 42)
    _, future_test = divide_groups(by[PERIODS[2]], 0.5, 43)   # same test half as study B
    return {"hist_train": hist_train, "hist_val": hist_val, "adapt_train": adapt_train,
            "adapt_val": adapt_val, "future_test": future_test, "future_all": by[PERIODS[2]]}


def build_cache(rows, path, workers):
    path = Path(path)
    done_flag = path.with_suffix(".done")
    if done_flag.exists():
        return np.load(path, mmap_mode="r")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(rows), 224, 224, 3))

    class Decode(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            with Image.open(rows[i]["local_path"]) as im:
                im.draft("RGB", (448, 448))
                im = im.convert("RGB").resize((224, 224), Image.BILINEAR)
            return i, torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())

    t = time.time()
    for n, (ids, ims) in enumerate(DataLoader(Decode(), batch_size=64, num_workers=workers)):
        arr[ids.numpy()] = ims.numpy()
        if n % 50 == 0:
            log(f"cache {int(ids[-1]) + 1}/{len(rows)} ({time.time() - t:.0f}s)")
    arr.flush()
    done_flag.write_text("ok")
    return np.load(path, mmap_mode="r")


def cloud_overlay(image, key, alpha, seed):
    """Fixed localized near-white overlay on a normalized tensor; same key -> same cloud."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
    noise = torch.rand((1, 1, 16, 16), generator=gen)
    smooth = F.avg_pool2d(noise, 5, stride=1, padding=2)
    mask = (smooth >= torch.quantile(smooth, 0.8)).float()
    mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
    white = ((1 - MEAN) / STD)
    return image * (1 - alpha * mask) + white * (alpha * mask)


class CachedImages(Dataset):
    def __init__(self, cache, rows, cloud_alpha=0.0, seed=42):
        self.cache, self.rows, self.alpha, self.seed = cache, rows, cloud_alpha, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = torch.from_numpy(np.array(self.cache[r["idx"]])).permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD
        if self.alpha > 0:
            x = cloud_overlay(x, r["local_path"], self.alpha, self.seed)
        return x, r["class_id"], i


def loader(cache, rows, bs, workers, alpha=0.0, shuffle=False, seed=42):
    return DataLoader(CachedImages(cache, rows, alpha), batch_size=bs, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, drop_last=shuffle,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


# ----------------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.enabled = True
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + self.up(self.down(x)) * self.scale
        return out


class Backbone(nn.Module):
    """DINOv2 with frozen early blocks run under no_grad and LoRA on the last K blocks."""

    def __init__(self, name, trainable_blocks=0, rank=8):
        super().__init__()
        from transformers import Dinov2Model
        self.model = Dinov2Model.from_pretrained(name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.width = self.model.config.hidden_size
        self.blocks = self.model.encoder.layer
        self.split = len(self.blocks) - trainable_blocks
        self.loras = []
        for block in list(self.blocks)[self.split:]:
            att = block.attention.attention
            for n in ("query", "key", "value"):
                m = LoRALinear(getattr(att, n), rank, 2 * rank)
                setattr(att, n, m)
                self.loras.append(m)
            for parent, n in ((block.attention.output, "dense"), (block.mlp, "fc1"), (block.mlp, "fc2")):
                m = LoRALinear(getattr(parent, n), rank, 2 * rank)
                setattr(parent, n, m)
                self.loras.append(m)

    def set_lora(self, on):
        for m in self.loras:
            m.enabled = on

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def _run(block, h):
        out = block(h)
        return out[0] if isinstance(out, (tuple, list)) else out

    def stem(self, x):
        with torch.no_grad():
            h = self.model.embeddings(x)
            for block in list(self.blocks)[:self.split]:
                h = self._run(block, h)
        return h

    def tail(self, h):
        for block in list(self.blocks)[self.split:]:
            h = self._run(block, h)
        return self.model.layernorm(h)

    def forward(self, x):
        return self.tail(self.stem(x))


def metrics(labels, preds, classes):
    labels, preds = np.asarray(labels), np.asarray(preds)
    ids = np.arange(classes)
    counts = np.bincount(preds, minlength=classes)
    return {"samples": int(len(labels)), "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, labels=ids, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
            "unique_predictions": int((counts > 0).sum()),
            "max_class_share": float(counts.max() / max(1, len(preds)))}


@torch.no_grad()
def evaluate(backbone, head, data, classes):
    backbone.eval()
    labels, preds = [], []
    for x, y, _ in data:
        with torch.autocast("cuda", dtype=torch.float16):
            tok = backbone(x.cuda(non_blocking=True))[:, 1:]
        preds.append(head(tok.float().mean(1)).argmax(1).cpu())
        labels.append(y)
    labels, preds = torch.cat(labels).numpy(), torch.cat(preds).numpy()
    return metrics(labels, preds, classes), preds


# ----------------------------------------------------------------------------- stages
def stage_features(args, cache, rows):
    out = args.output / "features.npz"
    if out.exists():
        return np.load(out)
    bb = Backbone(args.backbone).cuda().eval()
    data = loader(cache, rows, args.eval_batch, args.workers)
    means, clss = [], []
    t = time.time()
    with torch.no_grad():
        for n, (x, _, _) in enumerate(data):
            with torch.autocast("cuda", dtype=torch.float16):
                h = bb(x.cuda(non_blocking=True))
            means.append(h[:, 1:].float().mean(1).half().cpu())
            clss.append(h[:, 0].half().cpu())
            if n % 50 == 0:
                log(f"features {n * args.eval_batch}/{len(rows)} ({time.time() - t:.0f}s)")
    np.savez(out, mean=torch.cat(means).numpy(), cls=torch.cat(clss).numpy())
    log(f"features done: {len(rows) / (time.time() - t):.1f} img/s")
    del bb
    torch.cuda.empty_cache()
    return np.load(out)


def fit_head(x, y, classes, xv, yv, seed=42, epochs=300):
    torch.manual_seed(seed)
    x, xv = torch.tensor(x).float().cuda(), torch.tensor(xv).float().cuda()
    y, yv = torch.tensor(y).cuda(), torch.tensor(yv).cuda()
    counts = torch.bincount(y, minlength=classes).float().clamp_min(1)
    weight = (counts.sum() / (classes * counts))
    head = nn.Linear(x.shape[1], classes).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1, None
    for epoch in range(epochs):
        perm = torch.randperm(len(x), device="cuda")
        for i in range(0, len(x), 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(head(x[b]), y[b], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9:
            with torch.no_grad():
                score = f1_score(yv.cpu(), head(xv).argmax(1).cpu(), average="macro", zero_division=0)
            if score > best:
                best, best_state = score, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head.eval(), best


def stage_gate(args, feats, splits, classes):
    out = args.output / "gate.json"
    if out.exists():
        return json.loads(out.read_text()), torch.load(args.output / "head.pt", weights_only=True)
    f = feats["mean"].astype(np.float32)
    ix = {k: np.array([r["idx"] for r in v]) for k, v in splits.items()}
    lab = {k: np.array([r["class_id"] for r in v]) for k, v in splits.items()}
    head, val_f1 = fit_head(f[ix["hist_train"]], lab["hist_train"], classes,
                            f[ix["hist_val"]], lab["hist_val"])
    torch.save(head.state_dict(), args.output / "head.pt")
    scores = {}
    with torch.no_grad():
        for k in ("hist_val", "adapt_val", "adapt_train", "future_test", "future_all"):
            pred = head(torch.tensor(f[ix[k]]).cuda()).argmax(1).cpu().numpy()
            scores[k] = metrics(lab[k], pred, classes)
    # drift statistics on frozen features, 1000 per group
    drift = []
    try:
        from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
        rng = np.random.default_rng(42)
        def pick(name, n):
            return f[rng.choice(ix[name], size=min(n, len(ix[name])), replace=False)]
        hist_all = np.concatenate([ix["hist_train"], ix["hist_val"]])
        hsel = f[rng.choice(hist_all, size=min(2000, len(hist_all)), replace=False)]
        pairs = {"historical_vs_future": (hsel[:1000], pick("future_all", 1000)),
                 "historical_vs_adapt": (hsel[:1000], f[rng.choice(np.concatenate([ix["adapt_train"], ix["adapt_val"]]), 1000, replace=False)]),
                 "historical_split_half_null": (hsel[:1000], hsel[1000:2000])}
        for name, (a, b) in pairs.items():
            m = distribution_shift_metrics(a, b, permutations=args.permutations, permutation_limit=1000, seed=42)
            drift.append({"comparison": name, **{k: v for k, v in m.items() if not isinstance(v, (list, dict))}})
            log("drift", name, drift[-1].get("mmd_squared"), drift[-1].get("mmd_permutation_p"),
                drift[-1].get("domain_classifier_roc_auc"))
    except Exception as exc:  # keep the gate result even if drift metrics fail
        drift.append({"error": repr(exc)})
    # clustered bootstrap: historical-val vs future-test accuracy gap (independent samples)
    rng = np.random.default_rng(0)
    def cluster_acc(rows_, preds_, labels_):
        g = np.array([r["group"] for r in rows_])
        u = np.unique(g)
        return g, u, (preds_ == labels_).astype(float)
    with torch.no_grad():
        ph = head(torch.tensor(f[ix["hist_val"]]).cuda()).argmax(1).cpu().numpy()
        pf = head(torch.tensor(f[ix["future_test"]]).cuda()).argmax(1).cpu().numpy()
    gh, uh, ch = cluster_acc(splits["hist_val"], ph, lab["hist_val"])
    gf, uf, cf = cluster_acc(splits["future_test"], pf, lab["future_test"])
    gaps = []
    for _ in range(1000):
        sh = rng.choice(uh, len(uh)); sf = rng.choice(uf, len(uf))
        a = np.concatenate([ch[gh == s] for s in sh]); b = np.concatenate([cf[gf == s] for s in sf])
        gaps.append(a.mean() - b.mean())
    gate = {"backbone": args.backbone, "head_hist_val_macro_f1": val_f1, "scores": scores, "drift": drift,
            "accuracy_gap_hist_val_minus_future": {"mean": float(np.mean(gaps)),
                                                    "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]},
            "counts": {k: len(v) for k, v in splits.items()}}
    gap = gate["accuracy_gap_hist_val_minus_future"]
    gate["harm_gate_passed"] = bool(gap["mean"] >= 0.03 and gap["ci95"][0] > 0)
    dump(out, gate)
    return gate, head.state_dict()


def stage_calib(args, cache, splits, head_state, classes):
    out = args.output / "calib.json"
    if out.exists():
        return json.loads(out.read_text())
    bb = Backbone(args.backbone).cuda().eval()
    head = nn.Linear(bb.width, classes).cuda()
    head.load_state_dict(head_state)
    rows = []
    for a in (0.0, 0.2, 0.35, 0.5, 0.7, 0.85):
        m, _ = evaluate(bb, head, loader(cache, splits["hist_val"], args.eval_batch, args.workers, a), classes)
        rows.append({"alpha": a, **m})
        log("calib", a, m["macro_f1"])
    clean = rows[0]["macro_f1"]
    pick = min(rows[1:], key=lambda r: abs((clean - r["macro_f1"]) - 0.075))
    res = {"candidates": rows, "selected_alpha": pick["alpha"],
           "selected_drop": clean - pick["macro_f1"],
           "within_5_10_points": bool(0.05 <= clean - pick["macro_f1"] <= 0.10)}
    dump(out, res)
    del bb
    torch.cuda.empty_cache()
    return res


def anchor_weights(ref_tokens, head, labels, mode, gen):
    """Weights over patches summing to one. Saliency = true-class evidence of the frozen model."""
    n = ref_tokens.shape[1]
    if mode == "uniform":
        return torch.full(ref_tokens.shape[:2], 1.0 / n, device=ref_tokens.device)
    evidence = (ref_tokens * head.weight[labels][:, None, :]).sum(-1)
    k = round(0.4 * n)
    top = evidence.topk(k, dim=1).indices
    w = torch.zeros_like(evidence).scatter_(1, top, 1.0 / k)
    if mode == "shuffled":
        order = torch.rand(w.shape, generator=gen).argsort(1).to(w.device)
        w = w.gather(1, order)
    return w


def train_arm(args, cache, splits, head, classes, mode, lam, alpha, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    bb = Backbone(args.backbone, args.trainable_blocks, args.rank).cuda()
    params = bb.trainable()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator().manual_seed(seed + 7)
    # step-zero equivalence: adapters start at zero
    x0 = next(iter(loader(cache, splits["adapt_val"][:8], 8, 0, alpha)))[0].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        bb.set_lora(False); a0 = head(bb(x0)[:, 1:].float().mean(1))
        bb.set_lora(True); b0 = head(bb(x0)[:, 1:].float().mean(1))
    step0 = float((a0 - b0).abs().max())
    history = []
    accum = max(1, args.effective_batch // args.batch)
    for epoch in range(args.epochs):
        bb.train()
        data = loader(cache, splits["adapt_train"], args.batch, args.workers, alpha, shuffle=True, seed=seed + epoch)
        ce_sum = an_sum = n = 0
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(data):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                h = bb.stem(x)
                tok = bb.tail(h)[:, 1:].float()
                ce = F.cross_entropy(head(tok.mean(1)), y)
                anchor = torch.zeros((), device="cuda")
                if mode != "plain" and lam > 0:
                    with torch.no_grad():
                        bb.set_lora(False)
                        ref = bb.tail(h)[:, 1:].float()
                        bb.set_lora(True)
                        w = anchor_weights(ref, head, y, mode, gen)
                    anchor = (w * (tok - ref).square().mean(-1)).sum(1).mean()
                loss = (ce + lam * anchor) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            ce_sum += float(ce); an_sum += float(anchor); n += 1
        val, _ = evaluate(bb, head, loader(cache, splits["adapt_val"], args.eval_batch, args.workers, alpha), classes)
        history.append({"epoch": epoch + 1, "ce": ce_sum / n, "anchor_mse": an_sum / n, "adapt_val": val})
        log(f"  {mode} lam={lam} a={alpha} s={seed} ep{epoch + 1} ce={ce_sum / n:.3f} anchor={an_sum / n:.4f} val_f1={val['macro_f1']:.4f}")
    test, preds = evaluate(bb, head, loader(cache, splits["future_test"], args.eval_batch, args.workers, alpha), classes)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del bb, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"mode": mode, "lambda": lam, "cloud_alpha": alpha, "seed": seed, "step0_max_logit_diff": step0,
            "trainable_parameters": int(sum(p.numel() for p in params)), "history": history,
            "adapt_val": history[-1]["adapt_val"], "future_test": test, "predictions": preds.tolist(),
            "peak_gpu_gib": peak}


def stage_arms(args, cache, splits, head_state, classes, alpha_cloud):
    head = nn.Linear(1024 if "large" in args.backbone else 768, classes).cuda().eval()
    head.load_state_dict(head_state)
    for p in head.parameters():
        p.requires_grad = False
    arms_dir = args.output / "arms"
    arms_dir.mkdir(exist_ok=True)
    conditions = {"clean": 0.0, "cloud": alpha_cloud}
    # frozen reference scores per condition (and regime A: clean model on cloudy test is the clean arm re-scored below)
    frozen = {}
    bb = Backbone(args.backbone).cuda().eval()
    for cond, a in conditions.items():
        p = arms_dir / f"frozen_{cond}.json"
        if not p.exists():
            m, preds = evaluate(bb, head, loader(cache, splits["future_test"], args.eval_batch, args.workers, a), classes)
            dump(p, {"mode": "frozen", "condition": cond, "future_test": m, "predictions": preds.tolist()})
        frozen[cond] = json.loads(p.read_text())["future_test"]
        log("frozen", cond, frozen[cond]["macro_f1"])
    del bb
    torch.cuda.empty_cache()
    for seed in args.seeds:
        for cond, a in conditions.items():
            for mode in ("plain", "uniform", "shuffled", "saliency"):
                lams = [0.0] if mode == "plain" else args.lambdas
                for lam in lams:
                    p = arms_dir / f"{cond}_{mode}_l{lam}_s{seed}.json"
                    if p.exists():
                        continue
                    t = time.time()
                    res = train_arm(args, cache, splits, head, classes, mode, lam, a, seed)
                    res["condition"] = cond
                    res["seconds"] = time.time() - t
                    dump(p, res)
                    log(f"DONE {p.name}: test_f1={res['future_test']['macro_f1']:.4f} val_f1={res['adapt_val']['macro_f1']:.4f} "
                        f"peak={res['peak_gpu_gib']:.1f}GiB {res['seconds'] / 60:.1f}min")


def cluster_boot(a_pred, b_pred, labels, groups, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    diff = (a_pred == labels).astype(float) - (b_pred == labels).astype(float)
    u = np.unique(groups)
    by = {g: diff[groups == g] for g in u}
    stats = [np.concatenate([by[g] for g in rng.choice(u, len(u))]).mean() for _ in range(reps)]
    return {"delta_acc": float(diff.mean()), "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]}


def stage_summary(args, splits, classes):
    arms_dir = args.output / "arms"
    labels = np.array([r["class_id"] for r in splits["future_test"]])
    groups = np.array([r["group"] for r in splits["future_test"]])
    results = [json.loads(p.read_text()) for p in sorted(arms_dir.glob("*.json"))]
    table, chosen = [], {}
    for r in results:
        table.append({"condition": r.get("condition"), "mode": r["mode"], "lambda": r.get("lambda"),
                      "seed": r.get("seed"), "adapt_val_f1": r.get("adapt_val", {}).get("macro_f1"),
                      **{f"test_{k}": r["future_test"][k] for k in ("macro_f1", "accuracy", "unique_predictions")}})
    # lambda chosen per (condition, mode) on mean adapt-val F1 across seeds (never test)
    for cond in ("clean", "cloud"):
        for mode in ("uniform", "shuffled", "saliency"):
            cands = {}
            for r in results:
                if r.get("condition") == cond and r["mode"] == mode:
                    cands.setdefault(r["lambda"], []).append(r["adapt_val"]["macro_f1"])
            if cands:
                chosen[(cond, mode)] = max(cands, key=lambda l: np.mean(cands[l]))
    def get(cond, mode, seed):
        lam = 0.0 if mode == "plain" else chosen.get((cond, mode))
        for r in results:
            if r.get("condition") == cond and r["mode"] == mode and r.get("seed") == seed and r.get("lambda") == lam:
                return r
    comparisons, means = [], {}
    for cond in ("clean", "cloud"):
        for mode in ("plain", "uniform", "shuffled", "saliency"):
            vals = [get(cond, mode, s)["future_test"]["macro_f1"] for s in args.seeds if get(cond, mode, s)]
            if vals:
                means[f"{cond}/{mode}"] = {"lambda": 0.0 if mode == "plain" else chosen.get((cond, mode)),
                                           "seeds": len(vals), "mean_test_macro_f1": float(np.mean(vals)),
                                           "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
        for a, b in (("saliency", "uniform"), ("saliency", "shuffled"), ("saliency", "plain"), ("uniform", "plain")):
            for s in args.seeds:
                ra, rb = get(cond, a, s), get(cond, b, s)
                if ra and rb:
                    comparisons.append({"condition": cond, "a": a, "b": b, "seed": s,
                                        "delta_macro_f1": ra["future_test"]["macro_f1"] - rb["future_test"]["macro_f1"],
                                        **cluster_boot(np.array(ra["predictions"]), np.array(rb["predictions"]), labels, groups)})
    dump(args.output / "summary.json", {"chosen_lambda": {f"{k[0]}/{k[1]}": v for k, v in chosen.items()},
                                        "means": means, "paired": comparisons, "all_runs": table})
    with open(args.output / "summary.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(table[0].keys()))
        w.writeheader(); w.writerows(table)
    log("summary written", json.dumps(means, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("/mnt/sdd/FYP/data/fmow_v1.1/subset_7k_country_matched/manifest_downloaded.csv"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--backbone", default="facebook/dinov2-large")
    ap.add_argument("--stages", nargs="+", default=["cache", "features", "gate", "calib", "arms", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--effective-batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--trainable-blocks", type=int, default=6)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--permutations", type=int, default=499)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap every split")
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "config.json", {k: str(v) for k, v in vars(args).items()})
    rows, classes = load_rows(args.manifest)
    splits = make_splits(rows)
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    log("rows", len(rows), "classes", len(classes), {k: len(v) for k, v in splits.items()})
    cache = build_cache(rows, Path("/mnt/sdd/FYP/cache/fmow21k_u8_224.npy"), args.workers) if "cache" in args.stages or True else None
    feats = stage_features(args, cache, rows) if any(s in args.stages for s in ("features", "gate")) else None
    gate, head_state = stage_gate(args, feats, splits, len(classes)) if feats is not None else (None, None)
    if gate:
        log("GATE", json.dumps({k: gate[k] for k in ("harm_gate_passed", "accuracy_gap_hist_val_minus_future")}),
            {k: round(v["macro_f1"], 4) for k, v in gate["scores"].items()})
    calib = stage_calib(args, cache, splits, head_state, len(classes)) if "calib" in args.stages else None
    if "arms" in args.stages:
        stage_arms(args, cache, splits, head_state, len(classes), calib["selected_alpha"])
    if "summary" in args.stages:
        stage_summary(args, splits, len(classes))


if __name__ == "__main__":
    main()
