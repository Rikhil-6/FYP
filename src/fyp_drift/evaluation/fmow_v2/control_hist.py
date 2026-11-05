"""Control: do heads trained on 2010-13 data do as well on 2016-17 as heads trained on 2014-15 data?"""
import json, os, sys, types
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import f1_score
from fyp_drift.evaluation import fmow_dinov2l_study as study
from fyp_drift.evaluation import saliency_attention_study as sas
from fyp_drift.evaluation.fmow_v2.context import BACKBONE, WIDTH, load_context, pool_2x2

ctx = load_context()
OUT = Path(os.environ["FYP_RUNS"]) / "control_hist"; OUT.mkdir(exist_ok=True)
HIST = Path("/mnt/sdd/FYP/cache/adapter_tokens_v2p_hist"); HIST.mkdir(exist_ok=True)
V2P = Path(os.environ["FYP_TOKENS"]); EXTRA = Path(os.environ["FYP_TOKENS_EXTRA"])

def extract(split):
    path = HIST / f"clean_{split}.npy"
    if path.with_suffix(".done").exists():
        return np.load(path, mmap_mode="r")
    rows = ctx.splits[split]
    backbone = study.Backbone(BACKBONE).cuda().eval()
    cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(rows), 100, WIDTH))
    n = 0
    with torch.no_grad():
        for images, _, _ in study.image_loader(ctx.cache, rows, 64, 4, 0.0, cloud_region=None):
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = backbone(images.cuda(non_blocking=True))[:, 1:].float()
            cells = pool_2x2(tokens); cache[n:n + len(cells)] = cells.half().cpu().numpy(); n += len(cells)
    assert n == len(rows); cache.flush(); path.with_suffix(".done").write_text("ok")
    study.log("extracted", split, n)
    del backbone; torch.cuda.empty_cache()
    return np.load(path, mmap_mode="r")

lab = lambda s: np.array([r["class_id"] for r in ctx.splits[s]])
hist = {"hist_train": (extract("hist_train"), lab("hist_train")), "hist_val": (extract("hist_val"), lab("hist_val"))}
adapt = {s: (np.load(V2P / f"clean_{s}.npy", mmap_mode="r"), lab(s)) for s in ("adapt_train", "adapt_val")}
test_labels = lab("future_test")
tests = {"clean": V2P / "clean_future_test.npy", "overlay": V2P / "cloud_future_test.npy",
         "off_object": EXTRA / "off_object_future_test.npy", "on_object": EXTRA / "on_object_future_test.npy"}
tests = {k: np.load(v, mmap_mode="r") for k, v in tests.items()}
test_rows = ctx.splits["future_test"]
cloudy = np.array([float(r.get("cloud_cover", 0) or 0) >= 10 for r in test_rows])
# class-matched weights so that non-cloudy images mirror the class mix of the cloudy subset
K = ctx.classes; pc = np.bincount(test_labels[cloudy], minlength=K) / cloudy.sum()
pn = np.bincount(test_labels[~cloudy], minlength=K) / (~cloudy).sum()
w_non = np.where(pn[test_labels] > 0, pc[test_labels] / np.maximum(pn[test_labels], 1e-9), 0.0)
head_state = torch.load(Path(os.environ["FYP_RUNS"]) / "main/head.pt", weights_only=True)
args = types.SimpleNamespace(lr=1e-3, epochs=15, batch=128)
mf1 = lambda y, p, w=None: float(f1_score(y, p, labels=np.arange(K), average="macro", sample_weight=w, zero_division=0))
res = []
# frozen baseline on the same sets
head = torch.nn.Linear(WIDTH, K).cuda(); head.load_state_dict(head_state); head.eval()
frozen = {}
with torch.no_grad():
    for name, tok in tests.items():
        preds = np.concatenate([head(torch.from_numpy(np.asarray(tok[i:i + 512])).cuda().float().mean(1)).argmax(1).cpu().numpy() for i in range(0, len(tok), 512)])
        frozen[name] = preds
    p = frozen["clean"]
    frozen["_real"] = dict(cloudy=mf1(test_labels[cloudy], p[cloudy]), clean_all=mf1(test_labels, p), noncloudy=mf1(test_labels[~cloudy], p[~cloudy]),
                           noncloudy_classmatched=mf1(test_labels[~cloudy], p[~cloudy], w_non[~cloudy]))
res.append({"model": "frozen", "train": "none", "seed": 0, **{k: mf1(test_labels, v) for k, v in frozen.items() if k != "_real"}, "real": frozen["_real"]})
print(json.dumps(res[-1]), flush=True)
for train_name, data in (("hist_2010_13", {"adapt_train": hist["hist_train"], "adapt_val": hist["hist_val"]}), ("adapt_2014_15", adapt)):
    full = {**data, "future_test": (tests["clean"], test_labels)}
    for arm in ("attention", "gated", "gated_shuffled"):
        for seed in (42, 43, 44):
            f = OUT / f"{train_name}_{arm}_s{seed}.json"
            if f.exists():
                res.append(json.loads(f.read_text())); continue
            model, best = sas.train(arm, seed, head_state, WIDTH, K, full, 0.0, args)
            r = {"model": arm, "train": train_name, "seed": seed, "val_f1": best["val_f1"], "epoch": best["epoch"]}
            for name, tok in tests.items():
                preds = sas.infer(model, tok, test_labels)[0]
                r[name] = mf1(test_labels, preds)
                if name == "clean":
                    r["real"] = dict(cloudy=mf1(test_labels[cloudy], preds[cloudy]), noncloudy=mf1(test_labels[~cloudy], preds[~cloudy]),
                                     noncloudy_classmatched=mf1(test_labels[~cloudy], preds[~cloudy], w_non[~cloudy]))
            f.write_text(json.dumps(r)); res.append(r); print(json.dumps(r), flush=True)
(OUT / "all.json").write_text(json.dumps(res, indent=1))
print("ALL DONE", flush=True)
