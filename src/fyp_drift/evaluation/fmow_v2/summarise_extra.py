import json, os
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from fyp_drift.evaluation.fmow_v2.context import load_context
ctx = load_context(); R = Path(os.environ["FYP_RUNS"]); K = ctx.classes
rows = ctx.splits["future_test"]; y = np.array([r["class_id"] for r in rows])
cloudy = np.array([float(r.get("cloud_cover", 0) or 0) >= 10 for r in rows])
pc = np.bincount(y[cloudy], minlength=K) / cloudy.sum(); pn = np.bincount(y[~cloudy], minlength=K) / (~cloudy).sum()
w = np.where(pn[y] > 0, pc[y] / np.maximum(pn[y], 1e-9), 0.0)
f1 = lambda yy, p, sw=None: 100 * f1_score(yy, p, labels=np.arange(K), average="macro", sample_weight=sw, zero_division=0)
mean = lambda xs: float(np.mean(xs))
def summarize(paths, get):
    out = []
    for p in paths:
        j = json.load(open(p)); c, ov, off, on = get(j)
        out.append(dict(clean=f1(y, c), acc=100 * accuracy_score(y, c), bal=100 * balanced_accuracy_score(y, c), real=f1(y[cloudy], c[cloudy]),
                        matched=f1(y[~cloudy], c[~cloudy], w[~cloudy]), overlay=f1(y, ov) if ov is not None else None, off=f1(y, off) if off is not None else None,
                        on=f1(y, on) if on is not None else None, secs=j.get("train_seconds"), params=j.get("trainable_parameters"), epoch=j.get("best_epoch"),
                        scalars=j.get("scalars"), box=(j["tests"]["clean"].get("box_share_of_support") if "tests" in j else None)))
    return {k: (round(mean([o[k] for o in out]), 2) if out and isinstance(out[0][k], (int, float)) else out[0][k] if out else None) for k in out[0]} | {"n": len(out)}
res = {}
hy = lambda j: (np.array(j["tests"]["clean"]["predictions"]), np.array(j["tests"]["overlay"]["predictions"]), np.array(j["tests"]["off_object"]["predictions"]) if "off_object" in j["tests"] else None, np.array(j["tests"]["on_object"]["predictions"]) if "on_object" in j["tests"] else None)
for r in (1, 2):
    res[f"hybrid_r{r}"] = summarize(sorted(R.glob(f"hybrid/clean_soft_gated_r{r}_w8_b2_s4*.json")), hy)
lo = lambda j: tuple(np.array(j["extra"][c]["predictions"]) if c in j["extra"] else None for c in ("clean", "overlay", "off_object", "on_object"))
for name, pat in (("lora_r2", "lora_fair/clean_r2_lr0.0003_s4*.json"), ("lora_r8", "lora_fair/clean_r8_lr0.0003_s4*.json"), ("lora_r1b2", "lora_fair/clean_r1_lr0.0003_b2_s4*.json")):
    res[name] = summarize(sorted(R.glob(pat)), lo)
# graded cloud for both hybrids, using their overlay predictions (same recipe as mix.py: 10 random subsets per seed)
rng = np.random.default_rng(0); perms = [rng.permutation(len(y)) for _ in range(20)]
for r in (1, 2):
    row = {}
    for f in (0.25, 0.5, 0.75):
        v = []
        for p in sorted(R.glob(f"hybrid/clean_soft_gated_r{r}_w8_b2_s4*.json")):
            c, ov, _, _ = hy(json.load(open(p)))
            for pm in perms[:10]:
                q = c.copy(); idx = pm[: int(round(f * len(y)))]; q[idx] = ov[idx]; v.append(f1(y, q))
        row[str(f)] = round(mean(v), 1)
    res[f"hybrid_r{r}"]["graded"] = row
print(json.dumps(res, indent=1))
