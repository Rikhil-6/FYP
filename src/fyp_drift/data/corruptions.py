"""Deterministic image shifts grouped by plausible deployment source.

The original corruption suite remains available for exact reproduction. The
realistic suite separates acquisition, imaging-pipeline, viewpoint, and content
loss effects so results are not dominated by a square occluder.
"""

import hashlib
import io

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from fyp_drift.data.clear100 import MEAN, STD

LEGACY_SCENARIOS = ("saturation", "gaussian_noise", "combined")
REALISTIC_INDIVIDUAL_SCENARIOS = (
    "illumination",
    "white_balance",
    "sensor_noise",
    "defocus_blur",
    "motion_blur",
    "jpeg_compression",
    "resolution_loss",
    "viewpoint_shift",
    "occlusion",
)
REALISTIC_BENCHMARK_SCENARIOS = (
    "clean",
    *REALISTIC_INDIVIDUAL_SCENARIOS,
    "realistic_mix",
)
SCENARIOS = (
    "clean",
    *REALISTIC_INDIVIDUAL_SCENARIOS,
    *LEGACY_SCENARIOS,
    "realistic_mix",
)


def _seed(index, scenario, severity, seed):
    value = f"{seed}:{index}:{scenario}:{severity}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def scenario_components(scenario, index, severity, seed=42):
    """Return the actual sources used for a sample, for logging and panels."""
    if scenario != "realistic_mix":
        return () if scenario == "clean" else (scenario,)
    candidates = [
        name for name in REALISTIC_INDIVIDUAL_SCENARIOS if name != "occlusion"
    ]
    ordered = sorted(candidates, key=lambda name: _seed(index, name, severity, seed))
    chosen = ordered[:severity]
    # Occlusion is a minority event rather than a compulsory component.
    if _seed(index, "occlusion-presence", severity, seed) % 10 == 0:
        chosen.append("occlusion")
    return tuple(chosen)


def _motion_blur(tensor, size, orientation):
    kernel = tensor.new_zeros((size, size))
    if orientation == 0:
        kernel[size // 2, :] = 1
    elif orientation == 1:
        kernel[:, size // 2] = 1
    else:
        kernel.diagonal().fill_(1)
    kernel /= kernel.sum()
    weight = kernel.expand(tensor.shape[0], 1, size, size)
    return torch.nn.functional.conv2d(
        tensor[None], weight, padding=size // 2, groups=tensor.shape[0]
    )[0].clamp(0, 1)


def _jpeg_roundtrip(tensor, quality):
    buffer = io.BytesIO()
    F.to_pil_image(tensor).save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return F.to_tensor(decoded.convert("RGB"))


def _apply_source(tensor, scenario, severity, index, seed):
    generator = torch.Generator().manual_seed(_seed(index, scenario, severity, seed))
    if scenario == "clean":
        return tensor
    if scenario == "illumination":
        delta = (0.06, 0.12, 0.18)[severity - 1]
        direction = (
            -1 if _seed(index, "illumination-direction", severity, seed) % 2 else 1
        )
        tensor = F.adjust_brightness(tensor, 1 + direction * delta)
        return F.adjust_contrast(tensor, 1 + direction * delta * 0.6).clamp(0, 1)
    if scenario == "white_balance":
        delta = (0.04, 0.08, 0.12)[severity - 1]
        warm = _seed(index, "temperature-direction", severity, seed) % 2 == 0
        gains = tensor.new_tensor(
            (1 + delta, 1.0, 1 - delta) if warm else (1 - delta, 1.0, 1 + delta)
        )[:, None, None]
        return (tensor * gains).clamp(0, 1)
    if scenario == "sensor_noise":
        read_noise = (0.004, 0.008, 0.012)[severity - 1]
        shot_noise = (0.008, 0.015, 0.025)[severity - 1]
        standard_deviation = read_noise + shot_noise * tensor.clamp_min(0).sqrt()
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        return (tensor + standard_deviation * noise).clamp(0, 1)
    if scenario == "defocus_blur":
        size = (3, 5, 7)[severity - 1]
        sigma = (0.6, 1.0, 1.5)[severity - 1]
        return F.gaussian_blur(tensor, (size, size), (sigma, sigma))
    if scenario == "motion_blur":
        return _motion_blur(
            tensor,
            (3, 5, 7)[severity - 1],
            _seed(index, "motion-direction", severity, seed) % 3,
        )
    if scenario == "jpeg_compression":
        return _jpeg_roundtrip(tensor, (85, 70, 55)[severity - 1])
    if scenario == "resolution_loss":
        size = (192, 160, 128)[severity - 1]
        reduced = F.resize(
            tensor,
            (size, size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return F.resize(
            reduced,
            (224, 224),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    if scenario == "viewpoint_shift":
        magnitude = (4, 8, 12)[severity - 1]
        signs = _seed(index, "viewpoint-direction", severity, seed)
        translate = [
            magnitude if signs & 1 else -magnitude,
            magnitude if signs & 2 else -magnitude,
        ]
        angle = (1.0, 2.0, 3.0)[severity - 1] * (1 if signs & 4 else -1)
        fill = [float(value) for value in tensor.mean(dim=(1, 2))]
        return F.affine(
            tensor,
            angle=angle,
            translate=translate,
            scale=(0.98, 0.95, 0.92)[severity - 1],
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )
    if scenario == "occlusion":
        area = (0.015, 0.03, 0.05)[severity - 1]
        side = max(1, round((area * 224 * 224) ** 0.5))
        top = int(torch.randint(0, 225 - side, (1,), generator=generator))
        left = int(torch.randint(0, 225 - side, (1,), generator=generator))
        fill = tensor.mean(dim=(1, 2), keepdim=True)
        result = tensor.clone()
        result[:, top : top + side, left : left + side] = fill
        return result
    # Legacy sources preserve the completed experiment exactly.
    if scenario == "saturation":
        return F.adjust_saturation(tensor, (0.90, 0.80, 0.70)[severity - 1])
    if scenario == "gaussian_noise":
        sigma = (0.01, 0.02, 0.035)[severity - 1]
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        return (tensor + sigma * noise).clamp(0, 1)
    raise ValueError(f"Unsupported source {scenario!r}")


def drift_transform(scenario, severity, seed=42):
    """Return a deterministic transform with three progressive severities."""
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown drift scenario {scenario!r}; choose from {SCENARIOS}"
        )
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")

    def apply(image, index=0):
        tensor = F.to_tensor(F.resize(image.convert("RGB"), (224, 224)))
        if scenario == "combined":
            # Exact frozen definition used by the completed legacy experiment.
            generator = torch.Generator().manual_seed(
                _seed(index, scenario, severity, seed)
            )
            tensor = F.adjust_saturation(tensor, (0.90, 0.80, 0.70)[severity - 1])
            sigma = (0.01, 0.02, 0.035)[severity - 1]
            tensor = (
                tensor
                + sigma
                * torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            ).clamp(0, 1)
            area = (0.02, 0.05, 0.08)[severity - 1]
            side = max(1, round((area * 224 * 224) ** 0.5))
            top = int(torch.randint(0, 225 - side, (1,), generator=generator))
            left = int(torch.randint(0, 225 - side, (1,), generator=generator))
            tensor[:, top : top + side, left : left + side] = tensor.mean(
                dim=(1, 2), keepdim=True
            )
            return F.normalize(tensor, MEAN, STD)
        sources = scenario_components(scenario, index, severity, seed)
        for source in sources:
            tensor = _apply_source(tensor, source, severity, index, seed)
        return F.normalize(tensor, MEAN, STD)

    return apply


def mixed_drift_transform(severity, seed=42, clean_repeats=3):
    """Frozen legacy clean/saturation/noise/occlusion adaptation mixture."""
    severities = (severity,) if isinstance(severity, int) else tuple(severity)
    if not severities or any(level not in (1, 2, 3) for level in severities):
        raise ValueError("severity levels must be drawn from 1, 2, and 3")
    scenarios = ("clean",) * clean_repeats + (
        "saturation",
        "gaussian_noise",
        "occlusion",
    )
    transforms = {
        (name, level): drift_transform(name, level, seed)
        for name in set(scenarios)
        for level in severities
    }

    def apply(image, index=0):
        scenario = scenarios[
            _seed(index, "mixture-scenario", severities, seed) % len(scenarios)
        ]
        level = severities[
            _seed(index, "mixture-severity", scenario, seed) % len(severities)
        ]
        return transforms[scenario, level](image, index)

    return apply


def realistic_mixed_drift_transform(severity, seed=42):
    """25% clean and 75% individually shifted samples for adaptation.

    Occlusion represents 1/12 of the stream. Compound mixtures remain held out
    for final testing, preventing the evaluation transform from leaking into
    adaptation.
    """
    severities = (severity,) if isinstance(severity, int) else tuple(severity)
    if not severities or any(level not in (1, 2, 3) for level in severities):
        raise ValueError("severity levels must be drawn from 1, 2, and 3")
    scenarios = ("clean", "clean", "clean", *REALISTIC_INDIVIDUAL_SCENARIOS)
    transforms = {
        (name, level): drift_transform(name, level, seed)
        for name in set(scenarios)
        for level in severities
    }

    def apply(image, index=0):
        scenario = scenarios[
            _seed(index, "realistic-adapt-source", severities, seed) % len(scenarios)
        ]
        level = severities[
            _seed(index, "realistic-adapt-level", scenario, seed) % len(severities)
        ]
        return transforms[scenario, level](image, index)

    return apply
