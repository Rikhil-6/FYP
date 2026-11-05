"""Token-wise gated LoRA: the old model's saliency scales the LoRA update at each patch, so salient patches adapt and context patches stay close to the old features.

The saliency permission p_i = rho + (1 - rho) * sigmoid(gamma * z_i + delta) (the soft gate of the hybrid, learned jointly) multiplies the
low-rank update at every adapted matrix for patch i (the class token keeps gain 1). The pooling gate of the hybrid is kept, so this adds saliency
inside the backbone on top of the hybrid, with no new parameters.
Example::  python -m fyp_drift.evaluation.fmow_v2.gated_lora --rank 2 --blocks 2 --seed 42
"""

from __future__ import annotations

import argparse

import torch
from fyp_drift.evaluation import fmow_dinov2l_study as study
from fyp_drift.evaluation.fmow_v2 import hybrid
from fyp_drift.evaluation.fmow_v2.context import load_context, pool_2x2


def _forward(self, x):
    output = self.base(x)
    if not self.enabled:
        return output
    update = 2.0 * self.up(self.down(x))
    gain = getattr(self, "gain", None)
    return output + (update if gain is None else update * gain.to(update.dtype))


study.LoRALinear.forward = _forward


class GatedLoRAHybrid(hybrid.Hybrid):
    def forward(self, images):
        with torch.autocast("cuda", dtype=torch.float16):
            early = self.backbone.early_features(images)
        z = self.saliency(early)  # old model's evidence, unadapted tokens, no gradient
        pooling = self.pooling
        rho = torch.sigmoid(pooling.rho_logit)
        permission = rho + (1 - rho) * torch.sigmoid(pooling.gamma * z + pooling.delta)  # (B, 100)
        grid = permission.reshape(-1, 10, 10).repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(len(z), 400)
        gain = torch.cat([torch.ones_like(grid[:, :1]), grid], 1)[..., None]  # (B, 401, 1)
        for adapter in self.backbone.adapters:
            adapter.gain = gain
        with torch.autocast("cuda", dtype=torch.float16):
            cells = pool_2x2(self.backbone.late_features(early)[:, 1:].float())
        for adapter in self.backbone.adapters:
            adapter.gain = None
        logits, w = pooling(cells.float(), z)
        return logits, w, cells


hybrid.Hybrid = GatedLoRAHybrid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="cap every split (smoke test only)")
    args = parser.parse_args(argv)
    ctx = load_context(args.limit)
    out = ctx.runs / "hybrid"
    out.mkdir(exist_ok=True)
    path = out / f"clean_gated_lora_r{args.rank}_w8_b{args.blocks}_s{args.seed}.json"
    if path.exists() and not args.limit:
        study.log("exists, skipping", path.name)
        return
    result, _ = hybrid.train_and_score(ctx, "soft_gated", args.rank, args.blocks, args.seed)
    result["kind"] = "gated_lora"
    if not args.limit:
        study.write_json(path, result)
    study.log("done", path.name, {k: round(v["future_test"]["macro_f1"], 4) for k, v in result["tests"].items()})


if __name__ == "__main__":
    main()
