"""Aggregate the saved run outputs into one versioned results table (CPU only, no images needed).

Reads the per-seed prediction files in ``results/fmow_v2/run_outputs`` and the 2016-17 test labels in
``results/fmow_v2/test_labels.npz`` and writes

* ``results/fmow_v2/results_table.json``: every headline number (Tables 2, 4, 5 and 6 of the paper and the result
  figures), with the seeds used and a hash of every file read;
* ``results/fmow_v2/summary_tables.md``: the same numbers as a table. Neither file is edited by hand.

Example::  python -m fyp_drift.evaluation.fmow_v2.aggregate
"""

import argparse
import glob
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")  # rare classes may go unpredicted

SCHEMA = 2
FRACTIONS = (0.25, 0.5, 0.75)
BLOCK_WEIGHTS_M = 12.5829  # weights in the four linear layers of one ViT-L block (qkv, proj, fc1, fc2), millions
HEAD_PARAMS = {"attention": 196929, "hard_gate": 196931, "hard_gate_shuffled": 196931, "soft_gate": 196932}

# key, label, kind, source pattern(s) (relative to run_outputs; {s} is the seed), mix.json name, backbone blocks changed
MODELS = [
    dict(key="frozen", label="Frozen baseline", kind="frozen", mix="frozen", params=0, blocks=0),
    dict(key="attention", label="Learned attention", kind="head", mix="Learned attention", params=HEAD_PARAMS["attention"],
         clean="saliency_attention/clean_attention_s{s}.json", off="heldout_attn_off_object/clean_attention_s{s}.json",
         on="heldout_attn_on_object/clean_attention_s{s}.json"),
    dict(key="hard_gate", label="Hard saliency gate", kind="head", mix="Hard gate", params=HEAD_PARAMS["hard_gate"],
         clean="saliency_attention/clean_gated_s{s}.json", off="heldout_attn_off_object/clean_gated_s{s}.json",
         on="heldout_attn_on_object/clean_gated_s{s}.json"),
    dict(key="hard_gate_shuffled", label="Hard gate, shuffled twin", kind="head", mix="Hard gate, shuffled",
         params=HEAD_PARAMS["hard_gate_shuffled"], clean="saliency_attention/clean_gated_shuffled_s{s}.json",
         off="heldout_attn_off_object/clean_gated_shuffled_s{s}.json", on="heldout_attn_on_object/clean_gated_shuffled_s{s}.json"),
    dict(key="soft_gate", label="Soft saliency gate", kind="head", mix=None, params=HEAD_PARAMS["soft_gate"],
         clean="saliency_soft/clean_soft_gated_s{s}.json", off="heldout_soft_off_object/clean_soft_gated_s{s}.json",
         on="heldout_soft_on_object/clean_soft_gated_s{s}.json"),
    dict(key="soft_gate_shuffled", label="Soft gate, shuffled twin", kind="head", mix=None, params=HEAD_PARAMS["soft_gate"],
         clean="saliency_soft/clean_soft_gated_shuffled_s{s}.json", off="heldout_soft_off_object/clean_soft_gated_shuffled_s{s}.json",
         on="heldout_soft_on_object/clean_soft_gated_shuffled_s{s}.json"),
    dict(key="lora_r1b2", label="LoRA r1, last 2 blocks", kind="lora", mix="LoRA r1, last 2 blocks", blocks=2,
         pattern="lora_fair/clean_r1_lr0.0003_b2_s{s}.json"),
    dict(key="lora_r2b2", label="LoRA r2, last 2 blocks", kind="lora", mix=None, blocks=2,
         pattern="lora_fair/clean_r2_lr0.0003_b2_s{s}.json"),
    dict(key="hybrid_r1", label="Hybrid: LoRA r1 + soft gate", kind="hybrid", mix="Hybrid r1", blocks=2,
         pattern="hybrid/clean_soft_gated_r1_w8_b2_s{s}.json"),
    dict(key="hybrid_r2", label="Hybrid: LoRA r2 + soft gate", kind="hybrid", mix=None, blocks=2,
         pattern="hybrid/clean_soft_gated_r2_w8_b2_s{s}.json"),
    dict(key="hybrid_r2_nosal", label="Hybrid control: LoRA r2 + attention scorer (no saliency)", kind="hybrid", mix=None,
         blocks=2, pattern="hybrid/clean_attention_r2_w8_b2_s{s}.json"),
    dict(key="gated_lora_r2", label="Token-gated LoRA r2 + soft gate (saliency scales the LoRA update)", kind="hybrid", mix=None,
         blocks=2, pattern="hybrid/clean_gated_lora_r2_w8_b2_s{s}.json"),
    dict(key="lora_r2", label="LoRA r2, 6 blocks", kind="lora", mix="LoRA r2, 6 blocks", blocks=6,
         pattern="lora_fair/clean_r2_lr0.0003_s{s}.json"),
    dict(key="lora_r8", label="LoRA r8, 6 blocks", kind="lora", mix="LoRA r8, 6 blocks", blocks=6,
         pattern="lora_fair/clean_r8_lr0.0003_s{s}.json"),
]
CONDITIONS = ("clean", "overlay", "off_object", "on_object")


def _f1(y, p, weight=None):
    return 100 * f1_score(y, p, labels=np.arange(62), average="macro", sample_weight=weight, zero_division=0)


class Labels:
    """2016-17 test labels, cloud cover and the mask of test images whose site occurs in an earlier split."""

    def __init__(self, path):
        z = np.load(path)
        self.y = z["class_id"].astype(int)
        self.cloudy = z["cloud_cover"] >= 10
        self.seen = z["seen_site"].astype(bool) if "seen_site" in z else None
        k = 62
        p_cloudy = np.bincount(self.y[self.cloudy], minlength=k) / max(self.cloudy.sum(), 1)
        p_clear = np.bincount(self.y[~self.cloudy], minlength=k) / max((~self.cloudy).sum(), 1)
        self.weight = np.where(p_clear[self.y] > 0, p_cloudy[self.y] / np.maximum(p_clear[self.y], 1e-9), 0.0)
        rng = np.random.default_rng(0)
        self.perms = [rng.permutation(len(self.y)) for _ in range(20)]

    def scores(self, p):
        y, c = self.y, self.cloudy
        out = dict(clean=_f1(y, p), acc=100 * accuracy_score(y, p), bal=100 * balanced_accuracy_score(y, p),
                   real=_f1(y[c], p[c]), matched=_f1(y[~c], p[~c], self.weight[~c]))
        if self.seen is not None and (~self.seen).any():
            out["clean_unseen_sites"] = _f1(y[~self.seen], p[~self.seen])
        return out

    def graded(self, clean, overlay, fraction):
        """Macro-F1 when a random `fraction` of test images is replaced by its cloudy prediction (10 draws)."""
        vals = []
        for perm in self.perms[:10]:
            q = clean.copy()
            idx = perm[: int(round(fraction * len(self.y)))]
            q[idx] = overlay[idx]
            vals.append(_f1(self.y, q))
        return float(np.mean(vals))


def _read(runs, rel, used):
    path = Path(runs) / rel
    used[rel] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return json.loads(path.read_text())


def _seeds(runs, pattern):
    found = []
    for s in (42, 43, 44):
        if (Path(runs) / pattern.format(s=s)).exists():
            found.append(s)
    return found


def _predictions(spec, runs, seed, used):
    """Predictions per condition for one model and seed (None where a condition was not stored)."""
    arr = lambda x: None if x is None else np.array(x)
    if spec["kind"] == "head":
        g = lambda rel: arr(_read(runs, rel.format(s=seed), used)["predictions"]) if (Path(runs) / rel.format(s=seed)).exists() else None
        return dict(clean=g(spec["clean"]), off_object=g(spec["off"]), on_object=g(spec["on"]), overlay=None), {}
    j = _read(runs, spec["pattern"].format(s=seed), used)
    tests = j["tests"] if spec["kind"] == "hybrid" else j["extra"]
    preds = {c: arr(tests[c]["predictions"]) if c in tests else None for c in CONDITIONS}
    return preds, dict(params=j.get("trainable_parameters"), seconds=j.get("train_seconds"))


def build_table(runs, labels_path, git_commit=None):
    runs = Path(runs)
    L = Labels(labels_path)
    mix_path = runs / "control_hist/mix.json"
    used = {}
    mix = _read(runs, "control_hist/mix.json", used)["macro_f1"] if mix_path.exists() else {}
    rows = []
    for spec in MODELS:
        if spec["kind"] == "frozen":
            j = _read(runs, "main/arms/frozen_clean.json", used)
            u = _read(runs, "main/summary_unseen_cloud.json", used)["frozen"]
            per_seed = [(L.scores(np.array(j["predictions"])), dict(off_object=100 * u["off_object"]["macro_f1"],
                                                                     on_object=100 * u["on_object"]["macro_f1"]), {})]
            seeds = [0]
        else:
            pat = spec.get("pattern") or spec["clean"]
            seeds = _seeds(runs, pat)
            if not seeds:
                continue
            per_seed = []
            for s in seeds:
                preds, extra = _predictions(spec, runs, s, used)
                sc = L.scores(preds["clean"])
                cond = {c: _f1(L.y, preds[c]) for c in ("off_object", "on_object", "overlay") if preds[c] is not None}
                if preds["overlay"] is not None:
                    cond["graded"] = {str(f): L.graded(preds["clean"], preds["overlay"], f) for f in FRACTIONS}
                per_seed.append((sc, cond, extra))
        row = dict(key=spec["key"], label=spec["label"], kind=spec["kind"], n_seeds=len(seeds), seeds=seeds)
        mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None
        for k in ("clean", "acc", "bal", "real", "matched", "clean_unseen_sites"):
            row[k] = mean([sc[k] for sc, _, _ in per_seed if k in sc])
        row["beside"] = mean([c["off_object"] for _, c, _ in per_seed if "off_object" in c])
        row["over"] = mean([c["on_object"] for _, c, _ in per_seed if "on_object" in c])
        row["real_drop"] = None if row["real"] is None else round(row["matched"] - row["real"], 2)
        row["sd_clean"] = round(float(np.std([sc["clean"] for sc, _, _ in per_seed], ddof=1)), 2) if len(per_seed) > 1 else None
        graded = {}
        if per_seed and "graded" in per_seed[0][1]:
            graded = {k: mean([c["graded"][k] for _, c, _ in per_seed]) for k in map(str, FRACTIONS)}
            row["overlay"] = mean([c["overlay"] for _, c, _ in per_seed])
        elif spec.get("mix") and f"overlay|{spec['mix']}" in mix:
            m = mix[f"overlay|{spec['mix']}"]  # heads and frozen: retrained predictions are not stored per seed
            graded = {str(f): m[i] for i, f in enumerate((0.25, 0.5, 0.75), start=1)}
            row["overlay"] = m[-1]
        else:
            row["overlay"] = None
        row["graded"] = graded or None
        params = [e["params"] for _, _, e in per_seed if e.get("params")]
        secs = [e["seconds"] for _, _, e in per_seed if e.get("seconds")]
        row["trainable_params"] = int(params[0]) if params else spec.get("params")
        row["backbone_weights_changed_m"] = round(spec.get("blocks", 0) * BLOCK_WEIGHTS_M, 1)
        row["train_seconds"] = round(float(np.mean(secs))) if secs else None
        rows.append(row)
    return dict(schema_version=SCHEMA, git_commit=git_commit, test_images=int(len(L.y)), real_cloud_images=int(L.cloudy.sum()),
                test_images_from_seen_sites=None if L.seen is None else int(L.seen.sum()), fractions=list(FRACTIONS),
                models=rows, sources=dict(sorted(used.items())))


def to_markdown(table):
    f = lambda v, d=1: "–" if v is None else f"{v:.{d}f}"
    out = ["# Headline results (macro-F1 %, 2016–17 test)", "",
           "Generated by `python -m fyp_drift.evaluation.fmow_v2.aggregate` from `run_outputs/` (do not edit by hand). "
           "Each row is the mean over the seeds in the `n` column; the frozen baseline has no seed.", "",
           "| Model | n | Trainable params | Backbone weights changed | Clean | Accuracy | Beside | Over | Real clouds | Real, class-matched | "
           "Clean, unseen sites | 25% | 50% | 75% | 100% |", "|" + "---|" * 15]
    for r in table["models"]:
        g = r["graded"] or {}
        pc = "–" if r["trainable_params"] is None else f"{r['trainable_params']:,}"
        bw = "0" if not r["backbone_weights_changed_m"] else f"{r['backbone_weights_changed_m']:.1f} M"
        out.append(f"| {r['label']} | {r['n_seeds'] or 1} | {pc} | {bw} | {f(r['clean'])} | {f(r['acc'])} | {f(r['beside'])} | "
                   f"{f(r['over'])} | {f(r['real'])} | {f(r['matched'])} | {f(r['clean_unseen_sites'])} | {f(g.get('0.25'))} | "
                   f"{f(g.get('0.5'))} | {f(g.get('0.75'))} | {f(r['overlay'])} |")
    out += ["", "Beside / Over: held-out cumulus clouds placed beside or over the object. Real clouds: the "
            f"{table['real_cloud_images']} test images with cloud cover of at least 10%. Class-matched: cloud-free images weighted to "
            "the cloudy images' class mix (the gap to the real-cloud column is the drop not explained by class mix). "
            "Unseen sites: the test images whose site does not appear in any earlier split "
            f"({table['test_images'] - (table['test_images_from_seen_sites'] or 0)} of {table['test_images']}). "
            "25–100%: share of test images replaced by their cloud-overlay version.", ""]
    return "\n".join(out)


def main():
    root = Path(__file__).resolve().parents[4]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=root / "results/fmow_v2/run_outputs")
    ap.add_argument("--labels", type=Path, default=root / "results/fmow_v2/test_labels.npz")
    ap.add_argument("--out", type=Path, default=root / "results/fmow_v2")
    a = ap.parse_args()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    table = build_table(a.runs, a.labels, commit)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results_table.json").write_text(json.dumps(table, indent=1) + "\n")
    (a.out / "summary_tables.md").write_text(to_markdown(table))
    for r in table["models"]:
        print(f"{r['label']:<58} n={r['n_seeds']} clean {r['clean']} acc {r['acc']} beside {r['beside']} over {r['over']} real {r['real']}")


if __name__ == "__main__":
    main()

