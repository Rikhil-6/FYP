import numpy as np
import torch
from torch import nn

from fyp_drift.evaluation.fmow_dinov2l_study import add_cloud, anchor_weights, cluster_bootstrap, object_box_mask, realistic_cloud, split_by_group


def rows(n=60):
    return [{"group": f"g{i % 12}", "local_path": f"p{i}"} for i in range(n)]


def test_group_split_keeps_locations_disjoint():
    kept, held = split_by_group(rows(), 0.25, 42)
    assert not {r["group"] for r in kept} & {r["group"] for r in held}
    assert len(kept) + len(held) == 60


def test_cloud_is_deterministic_per_image_and_local():
    image = torch.zeros(3, 224, 224)
    first, again, other = add_cloud(image, "a", 0.5), add_cloud(image, "a", 0.5), add_cloud(image, "b", 0.5)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    changed = (first != image).any(0).float().mean()
    assert 0.05 < changed < 0.6


def test_anchor_modes_sum_to_one_and_differ():
    torch.manual_seed(0)
    tokens, head, labels = torch.randn(4, 256, 16), nn.Linear(16, 5), torch.tensor([0, 1, 2, 3])
    weights = {m: anchor_weights(tokens, head, labels, m, torch.Generator().manual_seed(1)) for m in ("uniform", "shuffled", "saliency")}
    for w in weights.values():
        assert torch.allclose(w.sum(1), torch.ones(4))
    assert (weights["saliency"] > 0).sum(1).eq(round(0.4 * 256)).all()
    assert (weights["shuffled"] > 0).sum(1).eq(round(0.4 * 256)).all()
    assert not torch.equal(weights["saliency"], weights["shuffled"])


def test_cluster_bootstrap_interval_contains_mean():
    values = np.random.default_rng(0).normal(0.02, 0.1, 500)
    groups = np.repeat(np.arange(50), 10)
    result = cluster_bootstrap(values, groups, repetitions=300)
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]


def scene():
    row = {"key": "train/x/x_1/x_1_1_rgb.jpg", "box_cx": 0.5, "box_cy": 0.5, "box_w": 0.2, "box_h": 0.2,
           "sun_elevation": 30.0, "sun_azimuth": 160.0}
    return torch.full((3, 224, 224), 0.4), row


def test_realistic_cloud_respects_the_object_box():
    pixels, row = scene()
    box = object_box_mask(row).bool()
    off = realistic_cloud(pixels, row, 1.0, "off_object")
    on = realistic_cloud(pixels, row, 1.0, "on_object")
    brighter_off = (off > pixels + 1e-4).any(0)
    brighter_on = (on > pixels + 1e-4).any(0)
    assert not brighter_off[box].any()
    assert not brighter_on[~box].any()
    assert brighter_off.float().mean() > 0.05


def test_realistic_cloud_is_deterministic_and_casts_a_shadow():
    pixels, row = scene()
    first, again = realistic_cloud(pixels, row, 1.0, "off_object"), realistic_cloud(pixels, row, 1.0, "off_object")
    assert torch.equal(first, again)
    assert (first < pixels - 1e-4).any()
