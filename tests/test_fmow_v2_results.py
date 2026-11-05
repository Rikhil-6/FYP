"""Checks for the fMoW v2 results pipeline: aggregation, committed tables, figure inputs, saved models, site audit.

All run on CPU without images or a GPU. Run with:  PYTHONPATH=src pytest tests/test_fmow_v2_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "figures/fmow_v2/plots"))

from fyp_drift.evaluation.fmow_v2 import aggregate, site_overlap  # noqa: E402

RUNS = ROOT / "results/fmow_v2/run_outputs"
LABELS = ROOT / "results/fmow_v2/test_labels.npz"
TABLE = ROOT / "results/fmow_v2/results_table.json"


def _macro(y, p):
    return 100 * f1_score(y, p, labels=np.arange(62), average="macro", zero_division=0)


def _fake_runs(tmp_path, n=620, seeds=(42, 43)):
    """A tiny run_outputs tree: a learned-attention head and a hybrid, with known predictions."""
    rng = np.random.default_rng(1)
    y = np.arange(n) % 62
    cloud = rng.uniform(0, 30, n).astype(np.float32)
    np.savez(tmp_path / "labels.npz", class_id=y, cloud_cover=cloud, seen_site=rng.random(n) < 0.1)
    runs = tmp_path / "runs"
    preds = {}
    for s in seeds:
        noisy = lambda k: np.where(rng.random(n) < k, y, rng.integers(0, 62, n))
        for rel, k in (("saliency_attention/clean_attention_s{s}.json", 0.6), ("heldout_attn_off_object/clean_attention_s{s}.json", 0.55),
                       ("heldout_attn_on_object/clean_attention_s{s}.json", 0.45)):
            p = noisy(k)
            preds[(rel, s)] = p
            f = runs / rel.format(s=s)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"predictions": p.tolist()}))
        tests = {c: {"predictions": noisy(k).tolist()} for c, k in (("clean", 0.65), ("overlay", 0.5), ("off_object", 0.6), ("on_object", 0.5))}
        f = runs / f"hybrid/clean_soft_gated_r2_w8_b2_s{s}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"trainable_parameters": 83988, "train_seconds": 1800 + s, "tests": tests}))
    frozen = runs / "main/arms/frozen_clean.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps({"predictions": np.where(rng.random(n) < 0.4, y, 0).tolist()}))
    (runs / "main/summary_unseen_cloud.json").write_text(json.dumps({"frozen": {"off_object": {"macro_f1": 0.4}, "on_object": {"macro_f1": 0.35}}}))
    return runs, tmp_path / "labels.npz", y, preds


def test_aggregate_matches_direct_computation(tmp_path):
    runs, labels, y, preds = _fake_runs(tmp_path)
    table = aggregate.build_table(runs, labels)
    rows = {r["key"]: r for r in table["models"]}
    assert set(rows) == {"frozen", "attention", "hybrid_r2"}  # models without files are skipped, not invented
    att = rows["attention"]
    assert att["n_seeds"] == 2 and att["seeds"] == [42, 43]
    want = np.mean([_macro(y, preds[("saliency_attention/clean_attention_s{s}.json", s)]) for s in (42, 43)])
    assert att["clean"] == pytest.approx(want, abs=0.01)
    assert rows["hybrid_r2"]["trainable_params"] == 83988 and rows["hybrid_r2"]["backbone_weights_changed_m"] == 25.2
    assert rows["hybrid_r2"]["graded"].keys() == {"0.25", "0.5", "0.75"}
    assert rows["frozen"]["beside"] == pytest.approx(40.0) and rows["frozen"]["over"] == pytest.approx(35.0)
    assert table["real_cloud_images"] == int((np.load(labels)["cloud_cover"] >= 10).sum())


def test_graded_exposure_runs_from_clean_to_overlay(tmp_path):
    runs, labels, y, _ = _fake_runs(tmp_path)
    L = aggregate.Labels(labels)
    clean, overlay = np.array(y), (np.array(y) + 1) % 62
    assert L.graded(clean, overlay, 0.0 + 1e-9) == pytest.approx(100.0)
    assert L.graded(clean, overlay, 1.0) == pytest.approx(0.0)
    assert L.graded(clean, overlay, 0.25) > L.graded(clean, overlay, 0.75)


def test_markdown_lists_every_model(tmp_path):
    runs, labels, *_ = _fake_runs(tmp_path)
    table = aggregate.build_table(runs, labels)
    md = aggregate.to_markdown(table)
    for r in table["models"]:
        assert r["label"] in md


@pytest.mark.skipif(not (TABLE.exists() and RUNS.exists() and LABELS.exists()), reason="published results not present")
def test_committed_table_matches_raw_runs():
    """The published results_table.json and summary_tables.md must be what the raw run outputs give."""
    rebuilt = aggregate.build_table(RUNS, LABELS)
    committed = json.loads(TABLE.read_text())
    assert [m["key"] for m in rebuilt["models"]] == [m["key"] for m in committed["models"]]
    for a, b in zip(rebuilt["models"], committed["models"]):
        for k in ("clean", "acc", "bal", "beside", "over", "real", "matched", "overlay", "trainable_params", "n_seeds"):
            assert a[k] == pytest.approx(b[k], abs=0.01), (a["key"], k)
    assert aggregate.to_markdown(committed) == (ROOT / "results/fmow_v2/summary_tables.md").read_text()


@pytest.mark.skipif(not TABLE.exists(), reason="results table not present")
def test_figure_inputs_come_from_the_table():
    import results_table as rt

    rows = rt.accuracy_rows()
    assert len(rows) >= 8
    for label, clean, beside, over, real, colour, marker in rows:
        assert all(0 < v < 100 for v in (clean, beside, over, real)), label
    eff = rt.efficiency_rows()
    assert all(p > 0 for _, p, _, f1, *_ in eff for f1 in (f1,))


@pytest.mark.skipif(not (ROOT / "models/fmow_v2").exists(), reason="saved models not present")
def test_saved_models_load_and_are_finite():
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    folders = sorted((ROOT / "models/fmow_v2").iterdir())
    assert folders
    for folder in folders:
        files = sorted(folder.glob("*.safetensors"))
        assert files, folder
        for f in files:
            tensors = safetensors_numpy.load_file(str(f))
            assert tensors and all(np.isfinite(t).all() for t in tensors.values()), f
        assert (folder / "config.json").exists(), folder


def test_site_overlap_counts_cross_period_sites():
    row = lambda site, group: {"local_path": f"/d/{site}/{site}_0_rgb.jpg", "group": group}
    splits = {"hist_train": [row("a_1", "X:1"), row("a_2", "X:2")], "hist_val": [row("a_3", "X:3")],
              "adapt_train": [row("b_1", "Y:1")], "adapt_val": [row("b_2", "Y:2")],
              "future_test": [row("a_1", "X:1"), row("c_9", "Z:1"), row("b_2", "Y:2")]}
    r = site_overlap.audit(splits)
    assert r["test_sites"] == 3 and r["test_sites_shared_with_any_earlier_split"] == 2
    assert r["test_images_from_shared_sites"] == 2 and r["test_groups_also_earlier"] == 2
    assert site_overlap.seen_site_mask(splits) == [True, False, True]

