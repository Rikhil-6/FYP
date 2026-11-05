"""Graded cloud exposure: mix clean and cloudy predictions per image at fraction f of the 2016-17 test set."""
import json, os, types
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import f1_score
from fyp_drift.evaluation import fmow_dinov2l_study as study
from fyp_drift.evaluation import saliency_attention_study as sas
from fyp_drift.evaluation.fmow_v2.context import WIDTH, load_context

ctx = load_context(); R = Path(os.environ["FYP_RUNS"]); V2P = Path(os.environ["FYP_TOKENS"]); EXTRA = Path(os.environ["FYP_TOKENS_EXTRA"])
K = ctx.classes
lab = lambda s: np.array([r["class_id"] for r in ctx.splits[s]])
y = lab("future_test")
tests = {"clean": V2P / "clean_future_test.npy", "overlay": V2P / "cloud_future_test.npy", "on_object": EXTRA / "on_object_future_test.npy"}
tok = {k: np.load(v, mmap_mode="r") for k, v in tests.items()}
adapt = {s: (np.load(V2P / f"clean_{s}.npy", mmap_mode="r"), lab(s)) for s in ("adapt_train", "adapt_val")}
head_state = torch.load(R / "main/head.pt", weights_only=True)
args = types.SimpleNamespace(lr=1e-3, epochs=15, batch=128)
P = {}  # model -> seed -> {condition: predictions}
# frozen
h = torch.nn.Linear(WIDTH, K).cuda(); h.load_state_dict(head_state); h.eval()
with torch.no_grad():
    P["frozen"] = {0: {c: np.concatenate([h(torch.from_numpy(np.asarray(t[i:i+512])).cuda().float().mean(1)).argmax(1).cpu().numpy() for i in range(0, len(t), 512)]) for c, t in tok.items()}}
# heads, retrained on clean 2014-15 with the paper's recipe
for arm, name in (("attention", "Learned attention"), ("gated", "Hard gate"), ("gated_shuffled", "Hard gate, shuffled")):
    for seed in (42, 43, 44):
        model, _ = sas.train(arm, seed, head_state, WIDTH, K, {**adapt, "future_test": (tok["clean"], y)}, 0.0, args)
        P.setdefault(name, {})[seed] = {c: sas.infer(model, t, y)[0] for c, t in tok.items()}
        print("trained", name, seed, flush=True)
# LoRA and hybrid from saved predictions
for label, pat in (("LoRA r2, 6 blocks", "lora_fair/clean_r2_lr0.0003_s{}.json"), ("LoRA r8, 6 blocks", "lora_fair/clean_r8_lr0.0003_s{}.json"), ("LoRA r1, last 2 blocks", "lora_fair/clean_r1_lr0.0003_b2_s{}.json")):
    for s in (42, 43, 44):
        e = json.load(open(R / pat.format(s)))["extra"]
        P.setdefault(label, {})[s] = {c: np.array(e[c]["predictions"]) for c in ("clean", "overlay", "on_object")}
for s in (42, 43, 44):
    t = json.load(open(R / f"hybrid/clean_soft_gated_r1_w8_b2_s{s}.json"))["tests"]
    P.setdefault("Hybrid r1", {})[s] = {c: np.array(t[c]["predictions"]) for c in ("clean", "overlay")}
    # on_object not stored for the hybrid file
fracs = [0, .25, .5, .75, 1.0]; rng = np.random.default_rng(0); perms = [rng.permutation(len(y)) for _ in range(20)]
f1 = lambda p: f1_score(y, p, labels=np.arange(K), average="macro", zero_division=0)
out = {}
for cond in ("overlay", "on_object"):
    for model, seeds in P.items():
        if any(cond not in v for v in seeds.values()): continue
        row = []
        for f in fracs:
            vals = []
            for v in seeds.values():
                for pm in perms[:10]:
                    p = v["clean"].copy(); idx = pm[: int(round(f * len(y)))]; p[idx] = v[cond][idx]; vals.append(f1(p))
            row.append(round(100 * float(np.mean(vals)), 1))
        out[f"{cond}|{model}"] = row
        print(cond, model, row, flush=True)
(R / "control_hist").mkdir(exist_ok=True); (R / "control_hist/mix.json").write_text(json.dumps({"fractions": fracs, "macro_f1": out}, indent=1))
print("MIX DONE", flush=True)
