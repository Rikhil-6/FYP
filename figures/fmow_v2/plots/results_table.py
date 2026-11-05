"""Read results/fmow_v2/results_table.json (written by fyp_drift.evaluation.fmow_v2.aggregate) for the result figures."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE.parents[2] / "results/fmow_v2/results_table.json"
STATS = HERE / "data/v2_stats.json"

# model key -> (colour, marker); the figures plot these models in this order
STYLE = {
    "frozen": ("#8C8C8C", "o"),
    "attention": ("#8172B3", "o"),
    "hard_gate": ("#C44E52", "o"),
    "soft_gate": ("#64B5CD", "o"),
    "lora_r1b2": ("#937860", "s"),
    "hybrid_r1": ("#DD8452", "D"),
    "hybrid_r2": ("#E3A857", "D"),
    "lora_r2": ("#55A868", "s"),
    "lora_r8": ("#2E7D4F", "s"),
}


def load(path=TABLE):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `python -m fyp_drift.evaluation.fmow_v2.aggregate` first")
    return {m["key"]: m for m in json.loads(path.read_text())["models"]}


def accuracy_rows(path=TABLE):
    """(label, clean, beside, over, real, colour, marker) for the grouped-bar figure."""
    t = load(path)
    return [(t[k]["label"], t[k]["clean"], t[k]["beside"], t[k]["over"], t[k]["real"], *STYLE[k]) for k in STYLE if k in t]


def efficiency_rows(path=TABLE, stats=STATS):
    """(label, trainable params, backbone weights changed in millions, clean macro-F1, colour, marker)."""
    t = load(path)
    rows = [("Attention / saliency heads, width 128", t["attention"]["trainable_params"], 0, t["attention"]["clean"], "#8172B3", "o")]
    cp = json.loads(Path(stats).read_text())["cloud_param"].get("clean/attention/h8/73807")
    if cp:
        rows.append(("Attention head, width 8", 73807, 0, cp["f1"][0], "#B4A7D6", "o"))
    for k in ("lora_r1b2", "hybrid_r1", "hybrid_r2", "lora_r2", "lora_r8"):
        if k in t:
            m = t[k]
            rows.append((m["label"].replace("last 2 blocks", "2 blocks"), m["trainable_params"], m["backbone_weights_changed_m"], m["clean"], *STYLE[k]))
    return rows

