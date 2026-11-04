"""Build data/v2_stats.json (every number the figures and paper use) from the saved run outputs.

Example::  python stats.py ../../../results/fmow_v2/run_outputs
"""

import collections
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    V = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parents[2] / "results/fmow_v2/run_outputs")

    def J(path):
        return json.loads(Path(V, path).read_text())

    S = (42, 43, 44)
    out = {}

    def ms(vals):
        a = np.array(vals) * 100
        return (round(a.mean(), 1), round(a.std(), 1), len(a))

    g = J("main/gate.json")
    out["gate"] = {k: v for k, v in g.items() if k != "scores"}
    out["scores"] = {
        k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in g["scores"].items()
    }
    out["calib"] = [(c["alpha"], round(c["macro_f1"], 4)) for c in J("main/calib.json")["candidates"]]
    for c in ("clean", "cloud"):
        out[f"frozen_{c}"] = J(f"main/arms/frozen_{c}.json")["future_test"]
    lora = {}
    for c in ("clean", "cloud"):
        lora[f"r8_{c}"] = ms([J(f"main/arms/{c}_plain_l0.0_s{s}.json")["future_test"]["macro_f1"] for s in S])
        lora[f"r8_{c}_acc"] = ms([J(f"main/arms/{c}_plain_l0.0_s{s}.json")["future_test"]["accuracy"] for s in S])
        lora[f"r2_{c}"] = ms([J(f"lora_rank_sweep/{c}_plain_r2_s{s}.json")["future_test"]["macro_f1"] for s in S])
        lora[f"r2_{c}_acc"] = ms([J(f"lora_rank_sweep/{c}_plain_r2_s{s}.json")["future_test"]["accuracy"] for s in S])
        lora[f"r1_{c}"] = ms([J(f"lora_rank_sweep/{c}_plain_r1_s42.json")["future_test"]["macro_f1"]])
        lora[f"r1_{c}_acc"] = ms([J(f"lora_rank_sweep/{c}_plain_r1_s42.json")["future_test"]["accuracy"]])
    lora["seconds_r8"] = [round(J(f"main/arms/clean_plain_l0.0_s{s}.json").get("seconds", 0)) for s in S]
    lora["seconds_r2"] = [round(J(f"lora_rank_sweep/clean_plain_r2_s{s}.json").get("seconds", 0)) for s in S]
    out["lora"] = lora
    heads = {}
    for f in glob.glob(f"{V}/saliency_attention/*_s*.json"):
        r = json.load(open(f))
        heads.setdefault((r["condition"], r["arm"]), []).append(r)
    for f in glob.glob(f"{V}/saliency_soft/*_s*.json"):
        r = json.load(open(f))
        heads.setdefault((r["condition"], r["arm"]), []).append(r)
    H = {}
    for (c, a), rs in sorted(heads.items()):

        def acc(k, rs=rs):
            return np.mean([r["future_test"]["accuracy"] - r["faithfulness"][k]["accuracy"] for r in rs]) * 100

        H[f"{c}/{a}"] = {
            "n": len(rs),
            "f1": ms([r["future_test"]["macro_f1"] for r in rs]),
            "acc": ms([r["future_test"]["accuracy"] for r in rs]),
            "flip": round(100 * np.mean([r["faithfulness"]["uniform"]["flip_rate"] for r in rs])),
            "del_top25": round(acc("delete_top_0.25"), 1),
            "del_rand25": round(acc("delete_random_0.25"), 1),
            "del_bot25": round(acc("delete_bottom_0.25"), 1),
            "focus": round(np.mean([r["grounding"]["box_mass_over_share"] for r in rs]), 2),
            "point": round(np.mean([r["grounding"]["pointing_game"] for r in rs]), 2),
            "cloud": (
                round(np.mean([r["grounding"].get("cloud_mass", np.nan) for r in rs]), 3),
                round(np.mean([r["grounding"].get("cloud_share", np.nan) for r in rs]), 3),
            ),
            "scalars": {
                k: round(np.mean([r.get("learned_scalars", r.get("scalars", {})).get(k, np.nan) for r in rs]), 3)
                for k in ("beta", "gamma", "delta", "floor", "mix")
            },
        }
    out["heads"] = H
    hm = J("extra_metrics/head_metrics.json")
    out["extra"] = {k: {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)} for k, v in hm.items()}
    cp = collections.defaultdict(list)
    for f in glob.glob(f"{V}/cloud_param/*_s*.json"):
        r = json.load(open(f))
        cp[r["condition"], r["kind"], r["hidden"], r["trainable_parameters"]].append(r)
    out["cloud_param"] = {
        f"{k[0]}/{k[1]}/h{k[2]}/{k[3]}": {
            "f1": ms([r["future_test"]["macro_f1"] for r in rs]),
            "acc": ms([r["future_test"]["accuracy"] for r in rs]),
            "focus": round(np.mean([r["grounding"]["box_mass_over_share"] for r in rs]), 2),
            "cloud_mass": round(np.mean([r["grounding"].get("cloud_mass", np.nan) for r in rs]), 3),
            "scal": {
                kk: round(np.mean([r["scalars"].get(kk, np.nan) for r in rs]), 2)
                for kk in ("kappa", "tau", "gamma", "delta")
            },
        }
        for k, rs in sorted(cp.items())
    }
    out["detector"] = J("cloud_param/detector.json")
    sh = collections.defaultdict(list)
    for f in glob.glob(f"{V}/shift_test/*_s*.json"):
        r = json.load(open(f))
        sh[r["condition"], r["arm"]].append(r)

    def g2(rs, v, k, sub="grounding"):
        return round(np.mean([r[v][sub][k] for r in rs]), 3)

    out["shift"] = {
        f"{c}/{a}": {
            "n": len(rs),
            "f1_c": ms([r["centred"]["scores"]["macro_f1"] for r in rs]),
            "f1_s": ms([r["shifted"]["scores"]["macro_f1"] for r in rs]),
            **{
                f"{k}_{v[0]}": g2(rs, v, k)
                for v in ("centred", "shifted")
                for k in (
                    "pointing_game",
                    "topk_precision",
                    "box_mass_minus_centre_prior",
                    "box_mass_over_share",
                    "top_patch_on_border",
                    "centre_prior_pointing",
                )
            },
            "salmap_point_s": g2(rs, "shifted", "pointing_game", "saliency_grounding"),
            "salmap_point_c": g2(rs, "centred", "pointing_game", "saliency_grounding"),
            "salmap_mcp_s": g2(rs, "shifted", "box_mass_minus_centre_prior", "saliency_grounding"),
        }
        for (c, a), rs in sorted(sh.items())
    }
    sl = collections.defaultdict(list)
    for f in glob.glob(f"{V}/saliency_learning/*_s*.json"):
        r = json.load(open(f))
        sl[r["condition"], r["mode"]].append(r)
    out["sal_learning"] = {
        f"{c}/{m}": {
            "n": len(rs),
            "f1": ms([r["future_test"]["macro_f1"] for r in rs]),
            "focus": round(np.mean([r["grounding"]["box_mass_over_share"] for r in rs]), 2),
            "point": round(np.mean([r["grounding"]["pointing_game"] for r in rs]), 2),
            "del_top25": round(
                100
                * np.mean(
                    [r["future_test"]["accuracy"] - r["faithfulness"]["delete_top_0.25"]["accuracy"] for r in rs]
                ),
                1,
            ),
            "flip": round(100 * np.mean([r["faithfulness"]["uniform"]["flip_rate"] for r in rs])),
            "shift_f1": ms([r["shifted"]["scores"]["macro_f1"] for r in rs if "shifted" in r]),
            "shift_point": (
                round(np.mean([r["shifted"]["grounding"]["pointing_game"] for r in rs if "shifted" in r]), 2)
                if any(("shifted" in r for r in rs))
                else None
            ),
            "shift_focus": (
                round(np.mean([r["shifted"]["grounding"]["box_mass_over_share"] for r in rs if "shifted" in r]), 2)
                if any(("shifted" in r for r in rs))
                else None
            ),
        }
        for (c, m), rs in sorted(sl.items())
    }
    ro = {}
    for c in ("clean", "cloud"):
        for a in ("gated", "input"):
            rs = [J(f"saliency_router/{c}_{a}_s{s}.json") for s in S]
            for p in ("combined", "confidence_only", "saliency_only"):
                P = [r["policies"][p] for r in rs]
                d = {
                    "routed": ms([x["routed"]["macro_f1"] for x in P]),
                    "upd": ms([r["updated"]["macro_f1"] for r in rs]),
                    "old": ms([r["historical"]["macro_f1"] for r in rs]),
                    "diff": ms(
                        [x["routed"]["macro_f1"] - r["updated"]["macro_f1"] for x, r in zip(P, rs, strict=True)]
                    ),
                    "cov": round(np.mean([x["coverage"] for x in P]), 3),
                    "kept": [round(x["retained_gain"], 2) for x in P],
                }
                if p == "combined":
                    d["dm"] = {
                        f"{k}/{m}": round(np.mean([x[f"deletion_margin_{k}"][m] for x in P]), 2)
                        for k in ("helpful_route", "harmful_route")
                        for m in ("updated_map", "historical_map")
                    }
                ro[f"{c}/{a}/{p}"] = d
            ro[f"{c}/{a}/sizes"] = (rs[0]["selection_images"], rs[0]["calibration_images"])
    out["router"] = ro
    uc = {}
    for f in glob.glob(f"{V}/main/unseen_cloud/*_s42.json"):
        r = json.load(open(f))
        uc[os.path.basename(f)] = {
            "clean": round(100 * r["future_test"]["macro_f1"], 1),
            **{k: round(100 * v["future_test"]["macro_f1"], 1) for k, v in r.get("shifted_tests", {}).items()},
        }
    for f in glob.glob(f"{V}/main/unseen_cloud/frozen_*.json"):
        uc[os.path.basename(f)] = round(100 * json.load(open(f))["future_test"]["macro_f1"], 1)
    out["unseen"] = uc
    if os.path.exists(f"{V}/main/calib_regions.json"):
        out["calib_regions"] = {
            k: (v["selected_alpha"], round(v["selected_drop"], 3)) for k, v in J("main/calib_regions.json").items()
        }
    out["tokens"] = J("token_dropping/summary.json")
    out["strict"] = J("strict_saliency/summary.json")
    out["screens"] = (
        [r for r in csv.DictReader(open(f"{V}/screens/summary.csv"))]
        if os.path.exists(f"{V}/screens/summary.csv")
        else None
    )
    (HERE / "data/v2_stats.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
