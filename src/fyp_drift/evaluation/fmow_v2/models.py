"""Save and load trained v2 models as safetensors + JSON config, and run fast inference on images.

Example::  predictor = Predictor.load("models/fmow_v2/soft_gate"); predictor.predict("site.jpg")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from fyp_drift.evaluation.fmow_dinov2l_study import MEAN, STD, Backbone
from fyp_drift.evaluation.fmow_v2.context import BACKBONE, WIDTH, pool_2x2
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn

IMAGE_SIZE, GRID = 280, 10
KINDS = ("historical", "attention", "gated", "soft_gated", "lora", "hybrid")


def save_model(path, kind, tensors, classes, **config):
    """Write ``model.safetensors`` (weights only, no pickle) and ``config.json`` to ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path / "model.safetensors"))
    config = {
        "kind": kind,
        "classes": list(classes),
        "image_size": IMAGE_SIZE,
        "grid": GRID,
        "backbone": BACKBONE,
        **config,
    }
    (path / "config.json").write_text(json.dumps(config, indent=1))


def adapter_state(backbone):
    """Only the LoRA tensors of a backbone (the frozen DINOv2 weights come from the Hugging Face hub)."""
    return {k: v for k, v in backbone.state_dict().items() if ".down." in k or ".up." in k}


def build_head(config, tensors):
    """Rebuild a pooling head from its config and weights."""
    from fyp_drift.evaluation import saliency_attention_study as sas
    from fyp_drift.evaluation import soft_saliency_heads as soft
    from fyp_drift.evaluation.fmow_v2.hybrid import GatedPooling

    kind, classes = config["kind"], len(config["classes"])
    frozen = (
        {"weight": tensors["frozen.weight"], "bias": tensors["frozen.bias"]} if "frozen.weight" in tensors else None
    )
    if kind == "historical":
        head = nn.Linear(WIDTH, classes)
        head.load_state_dict({"weight": tensors["weight"], "bias": tensors["bias"]})
        return head
    if kind in ("attention", "gated"):
        head = sas.SaliencyAttentionHead(kind, frozen, WIDTH, classes)
    elif kind == "soft_gated":
        head = soft.SoftSaliencyHead(kind, frozen, WIDTH, classes)
    else:  # lora / hybrid pooling on adapted tokens
        historical = nn.Linear(WIDTH, classes)
        historical.load_state_dict(
            {"weight": tensors["pooling.classifier.weight"], "bias": tensors["pooling.classifier.bias"]}
        )
        head = GatedPooling(historical, config.get("pooling", "soft_gated"))
        tensors = {k.removeprefix("pooling."): v for k, v in tensors.items() if k.startswith("pooling.")}
    head.load_state_dict(tensors)
    return head


def load_image(path, box=None):
    """Crop to twice the object's longer side (as in training) when ``box`` = (cx, cy, w, h) in [0, 1] is given."""
    image = Image.open(path).convert("RGB")
    if box is not None:
        width, height = image.size
        cx, cy, bw, bh = box[0] * width, box[1] * height, box[2] * width, box[3] * height
        side = min(max(2 * max(bw, bh), 0.25 * min(width, height)), min(width, height))
        x0, y0 = min(max(cx - side / 2, 0), width - side), min(max(cy - side / 2, 0), height - side)
        image = image.crop((x0, y0, x0 + side, y0 + side))
    pixels = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC), dtype=np.float32) / 255.0
    return ((torch.from_numpy(pixels).permute(2, 0, 1) - MEAN) / STD).unsqueeze(0)


class Predictor:
    """A frozen (or LoRA-adapted) DINOv2-L plus a saved head, ready for inference."""

    def __init__(self, config, backbone, head, historical):
        self.config, self.backbone, self.head, self.historical = config, backbone, head, historical
        self.classes = config["classes"]

    @classmethod
    def load(cls, path, device=None):
        path = Path(path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        config = json.loads((path / "config.json").read_text())
        tensors = load_file(str(path / "model.safetensors"))
        blocks, rank = config.get("blocks", 0), config.get("rank", 1)
        backbone = Backbone(config["backbone"], blocks, rank)
        if blocks:
            backbone.load_state_dict({k: v for k, v in tensors.items() if ".down." in k or ".up." in k}, strict=False)
        head = build_head(config, tensors) if config["kind"] != "lora" else None
        historical = nn.Linear(WIDTH, len(config["classes"]))
        hist_keys = (
            ("frozen.weight", "frozen.bias") if "frozen.weight" in tensors else ("historical.weight", "historical.bias")
        )
        if hist_keys[0] in tensors:
            historical.load_state_dict({"weight": tensors[hist_keys[0]], "bias": tensors[hist_keys[1]]})
        elif config["kind"] == "historical":
            historical = head
        modules = [m for m in (backbone, head, historical) if m is not None]
        for module in modules:
            module.to(device).eval()
        return cls(config, backbone, head, historical)

    @torch.no_grad()
    def predict(self, image_path, box=None, top=5):
        """Top classes with probabilities and the 10 x 10 attention weights (None for heads without pooling)."""
        device = next(self.backbone.parameters()).device
        images = load_image(image_path, box).to(device)
        tokens = self.backbone(images)[:, 1:].float()
        cells = pool_2x2(tokens)
        kind, weights = self.config["kind"], None
        if kind == "historical":
            logits = self.head(cells.mean(1))
        elif kind == "lora":
            logits = self.historical(tokens.mean(1))
        elif kind == "hybrid":
            reference = self.saliency_reference(images)
            logits, weights = self.head(cells, reference)
        else:
            logits, weights, _ = self.head(cells, torch.Generator().manual_seed(7))
        probs = logits.softmax(1)[0]
        values, indices = probs.topk(top)
        result = {"top": [(self.classes[i], float(p)) for p, i in zip(values, indices, strict=True)]}
        if weights is not None:
            result["weights"] = weights[0].reshape(GRID, GRID).cpu().numpy()
        return result

    def saliency_reference(self, images):
        """Old model's standardised evidence on the unadapted tokens (the hybrid's gate input)."""
        self.backbone.use_adapters(False)
        reference = pool_2x2(self.backbone(images)[:, 1:].float())
        self.backbone.use_adapters(True)
        predicted = self.historical(reference.mean(1)).argmax(1)
        evidence = (reference * self.historical.weight[predicted][:, None, :]).sum(-1)
        return (evidence - evidence.mean(1, keepdim=True)) / evidence.std(1, keepdim=True).clamp_min(1e-6)
