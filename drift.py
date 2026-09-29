"""Deterministic, mild image corruptions for temporal robustness evaluation."""
import hashlib

import torch
from torchvision.transforms import functional as F

from data import MEAN, STD

SCENARIOS = ("clean", "saturation", "gaussian_noise", "occlusion", "combined")
ADAPT_SCENARIOS = ("clean", "clean", "clean", "saturation", "gaussian_noise", "occlusion")


def _seed(index, scenario, severity, seed):
    value = f"{seed}:{index}:{scenario}:{severity}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def drift_transform(scenario, severity, seed=42):
    """Return a deterministic transform. Severities 1-3 remain visually mild."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown drift scenario {scenario!r}; choose from {SCENARIOS}")
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")

    def apply(image, index=0):
        image = F.resize(image.convert("RGB"), (224, 224))
        generator = torch.Generator().manual_seed(_seed(index, scenario, severity, seed))
        tensor = F.to_tensor(image)
        if scenario in ("saturation", "combined"):
            factor = (0.90, 0.80, 0.70)[severity - 1]
            tensor = F.adjust_saturation(tensor, factor)
        if scenario in ("gaussian_noise", "combined"):
            sigma = (0.01, 0.02, 0.035)[severity - 1]
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            tensor = (tensor + sigma * noise).clamp(0, 1)
        if scenario in ("occlusion", "combined"):
            area = (0.02, 0.05, 0.08)[severity - 1]
            side = max(1, int(round((area * 224 * 224) ** 0.5)))
            top = int(torch.randint(0, 225 - side, (1,), generator=generator))
            left = int(torch.randint(0, 225 - side, (1,), generator=generator))
            # A local mean patch resembles loss of visible content without an
            # artificial high-contrast adversarial marker.
            fill = tensor.mean(dim=(1, 2), keepdim=True)
            tensor[:, top:top + side, left:left + side] = fill
        return F.normalize(tensor, MEAN, STD)

    return apply


def mixed_drift_transform(severity, seed=42, clean_repeats=3):
    """Deterministic clean/single-corruption mixture; combined stays held out.

    ``severity`` may be one level or a tuple of levels.  The enhanced adaptation
    profile uses every individual severity while retaining 25% clean examples.
    """
    severities = (severity,) if isinstance(severity, int) else tuple(severity)
    if not severities or any(level not in (1, 2, 3) for level in severities):
        raise ValueError("severity levels must be drawn from 1, 2, and 3")
    scenarios = ("clean",) * clean_repeats + ("saturation", "gaussian_noise", "occlusion")
    transforms = {(name, level): drift_transform(name, level, seed)
                  for name in set(scenarios) for level in severities}

    def apply(image, index=0):
        scenario = scenarios[_seed(index, "mixture-scenario", severities, seed) % len(scenarios)]
        level = severities[_seed(index, "mixture-severity", scenario, seed) % len(severities)]
        return transforms[scenario, level](image, index)

    return apply
