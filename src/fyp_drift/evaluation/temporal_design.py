"""Temporal-by-corruption experiment grids with unconfounded severity."""

from __future__ import annotations


def crossed_corruption_grid(periods, scenarios, severities=(1, 2, 3)):
    """Return clean and fixed-severity cells for every period and source.

    Severity is crossed with time: every period receives the same list of
    source/severity conditions. ``realistic_mix`` is rejected because its
    existing transform changes both severity and the number of sources.
    """
    periods = tuple(periods)
    scenarios = tuple(scenarios)
    severities = tuple(severities)
    if not periods or len(set(periods)) != len(periods):
        raise ValueError("periods must be a non-empty sequence without duplicates")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be a non-empty sequence without duplicates")
    if "clean" in scenarios:
        raise ValueError("clean is added automatically; omit it from scenarios")
    if "realistic_mix" in scenarios:
        raise ValueError(
            "realistic_mix changes source count with severity; evaluate it as a "
            "separate progression or define a fixed-source mixture"
        )
    if not severities or any(level not in (1, 2, 3) for level in severities):
        raise ValueError("severities must be selected from 1, 2, and 3")
    if len(set(severities)) != len(severities):
        raise ValueError("severities must not contain duplicates")

    cells = []
    for period in periods:
        cells.append({"period": period, "scenario": "clean", "severity": 0})
        for scenario in scenarios:
            for severity in severities:
                cells.append(
                    {"period": period, "scenario": scenario, "severity": severity}
                )
    return cells


def progressive_corruption_schedule(periods, scenario, severities):
    """Create a named progressive stress schedule, separate from the grid."""
    periods, severities = tuple(periods), tuple(severities)
    if not periods or len(periods) != len(severities):
        raise ValueError("one severity is required for each temporal period")
    if scenario == "realistic_mix":
        raise ValueError("use a fixed-source mixture to isolate severity progression")
    if any(level not in (0, 1, 2, 3) for level in severities):
        raise ValueError("progressive severities must be 0, 1, 2, or 3")
    if any(a > b for a, b in zip(severities, severities[1:])):
        raise ValueError("progressive severities must be non-decreasing")
    return [
        {
            "period": period,
            "scenario": "clean" if severity == 0 else scenario,
            "severity": severity,
        }
        for period, severity in zip(periods, severities)
    ]
