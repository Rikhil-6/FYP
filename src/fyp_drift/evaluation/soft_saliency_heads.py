"""Softer saliency heads: let surrounding context through instead of locking attention onto the object.

The gated and prior heads in saliency_attention_study made the frozen DINOv2-L attend to the labelled object
(pointing 0.57-0.62) but lost 2-3.5 macro-F1 points, consistent with fMoW classes depending on context. These
three designs keep saliency in the attention path but bound how much it can suppress:

  soft_gated  a ~ softmax(s) * (rho + (1 - rho) * sigmoid(gamma z + delta)), floor rho = sigmoid(r), starts at 0.5
              every patch keeps at least rho of its learned weight, so context cannot be switched off
  mix         a = (1 - lam) softmax(s) + lam softmax(beta z), lam = sigmoid(l), starts at 0.3
              a learned blend of plain attention and the saliency map
  soft_prior  a = softmax(beta z + s), beta starts at 0.25 and the scorer is NOT zero-initialised
              saliency nudges the learned scores rather than defining them

Each has a *_shuffled twin (z permuted within the image). Training, selection, faithfulness and grounding are
identical to saliency_attention_study, and every run is compared image by image with learned attention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fyp_drift.evaluation.fmow_dinov2l_study import DEFAULT_MANIFEST, cluster_bootstrap, load_rows, log, scores, study_splits, write_json
from fyp_drift.evaluation.saliency_adapters import SPLITS, batches
from fyp_drift.evaluation.saliency_attention_study import SaliencyAttentionHead, faithfulness, grounding, infer

SOFT = ("soft_gated", "mix", "soft_prior")
ARMS = SOFT + tuple(f"{a}_shuffled" for a in SOFT)
RUNS = Path("/mnt/sdd/FYP/runs/today")


class SoftSaliencyHead(SaliencyAttentionHead):
    def __init__(self, arm, head_state, width, classes):
        super().__init__(arm, head_state, width, classes)
        if self.kind == "soft_gated":
            self.gamma, self.delta, self.floor_raw = nn.Parameter(torch.tensor(2.0)), nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))
        if self.kind == "mix":
            self.beta, self.mix_raw = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(-0.85))
        if self.kind == "soft_prior":
            self.beta = nn.Parameter(torch.tensor(0.25))

    def weights(self, tokens, z):
        learned = self.scorer(tokens).squeeze(-1)
        if self.kind == "soft_gated":
            floor = torch.sigmoid(self.floor_raw)
            w = learned.softmax(1) * (floor + (1 - floor) * torch.sigmoid(self.gamma * z + self.delta))
            return w / w.sum(1, keepdim=True).clamp_min(1e-8)
        if self.kind == "mix":
            lam = torch.sigmoid(self.mix_raw)
            return (1 - lam) * learned.softmax(1) + lam * (self.beta * z).softmax(1)
        return (self.beta * z + learned).softmax(1)

    def scalars(self):
        out = {}
        if hasattr(self, "floor_raw"):
            out["floor"] = float(torch.sigmoid(self.floor_raw))
        if hasattr(self, "mix_raw"):
            out["mix"] = float(torch.sigmoid(self.mix_raw))
        for name in ("beta", "gamma", "delta"):
            if hasattr(self, name):
                out[name] = float(getattr(self, name))
        return out


def train(arm, seed, head_state, width, classes, data, args):
    torch.manual_seed(seed)
    model = SoftSaliencyHead(arm, head_state, width, classes).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    best = {"val_f1": -1.0}
    for epoch in range(args.epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        for tokens, targets in batches(*data["adapt_train"], args.batch, shuffle_seed=seed * 1000 + epoch):
            logits, _, _ = model(tokens, generator)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = scores(data["adapt_val"][1], infer(model, *data["adapt_val"])[0], classes)["macro_f1"]
        if val > best["val_f1"]:
            best = {"val_f1": val, "epoch": epoch + 1, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model.eval(), best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=RUNS / "saliency_soft")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    test_rows = splits["future_test"]
    labels = np.array([r["class_id"] for r in test_rows])
    groups = np.array([r["group"] for r in test_rows])
    head_state = torch.load(RUNS / "main/head.pt", weights_only=True)
    width = head_state["weight"].shape[1]
    for condition in ("clean", "cloud"):
        data = {s: (np.load(RUNS.parent.parent / "cache/adapter_tokens" / f"{condition}_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in splits[s]])) for s in SPLITS}
        for seed in args.seeds:
            for arm in ARMS:
                path = args.output / f"{condition}_{arm}_s{seed}.json"
                if path.exists():
                    continue
                model, best = train(arm, seed, head_state, width, classes, data, args)
                predictions, weights, _ = infer(model, *data["future_test"])
                result = {"arm": arm, "condition": condition, "seed": seed, "best_epoch": best["epoch"], "adapt_val_macro_f1": best["val_f1"],
                          "future_test": scores(labels, predictions, classes), "predictions": predictions.tolist(),
                          "faithfulness": faithfulness(model, *data["future_test"], weights, predictions),
                          "grounding": grounding(weights, test_rows, condition == "cloud"), "learned_scalars": model.scalars()}
                write_json(path, result)
                log(f"{condition} {arm} s{seed}: test={result['future_test']['macro_f1']:.4f} point={result['grounding']['pointing_game']:.3f} "
                    f"box={result['grounding']['box_mass_over_share']:.2f}x scalars={result['learned_scalars']}")
    summarize(args, labels, groups)


def summarize(args, labels, groups):
    runs = [json.loads(p.read_text()) for p in sorted(args.output.glob("*_s*.json"))]
    table, paired = {}, []
    for condition in ("clean", "cloud"):
        for arm in ARMS:
            rs = [r for r in runs if (r["condition"], r["arm"]) == (condition, arm)]
            if not rs:
                continue
            mean = lambda f: float(np.mean([f(r) for r in rs]))
            table[f"{condition}/{arm}"] = {"seeds": len(rs), "test_macro_f1": mean(lambda r: r["future_test"]["macro_f1"]),
                                           "pointing_game": mean(lambda r: r["grounding"]["pointing_game"]), "box_mass_over_share": mean(lambda r: r["grounding"]["box_mass_over_share"]),
                                           "uniform_flip": mean(lambda r: r["faithfulness"]["uniform"]["flip_rate"]),
                                           **({"cloud_mass": mean(lambda r: r["grounding"]["cloud_mass"]), "cloud_share": mean(lambda r: r["grounding"]["cloud_share"])} if condition == "cloud" else {}),
                                           "scalars": {k: float(np.mean([r["learned_scalars"][k] for r in rs])) for k in rs[0]["learned_scalars"]}}
            for r in rs:
                for reference in ("attention", "gated"):
                    path = RUNS / "saliency_attention" / f"{condition}_{reference}_s{r['seed']}.json"
                    if path.exists():
                        base = json.loads(path.read_text())
                        difference = (np.array(r["predictions"]) == labels).astype(float) - (np.array(base["predictions"]) == labels).astype(float)
                        paired.append({"condition": condition, "arm": arm, "reference": reference, "seed": r["seed"],
                                       "delta_macro_f1": r["future_test"]["macro_f1"] - base["future_test"]["macro_f1"],
                                       "delta_accuracy": cluster_bootstrap(difference, groups, repetitions=1000)})
    write_json(args.output / "summary.json", {"table": table, "paired": paired})
    log("summary", json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
