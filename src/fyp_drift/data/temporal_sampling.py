"""Deterministic sampling for matched chronological image subsets."""

from __future__ import annotations

import random
from collections import defaultdict


def matched_temporal_sample(
    rows,
    *,
    period_key,
    strata_keys,
    target_per_period,
    seed=42,
    group_key=None,
):
    """Select the same stratum mix and exact size from every temporal period.

    ``strata_keys`` commonly contains class and country fields. The quota for
    each stratum is limited by its smallest period count, then allocated
    proportionally with Hamilton's largest-remainder method. This keeps class
    and geography composition identical across periods while returning exactly
    ``target_per_period`` rows from each one.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("rows must not be empty")
    if target_per_period < 1:
        raise ValueError("target_per_period must be positive")
    strata_keys = tuple(strata_keys)
    if not strata_keys:
        raise ValueError("strata_keys must not be empty")

    periods = sorted({row.get(period_key) for row in rows}, key=str)
    if len(periods) < 2 or None in periods:
        raise ValueError("at least two non-missing temporal periods are required")

    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        period = row.get(period_key)
        if period is None:
            continue
        stratum = tuple(row.get(key) for key in strata_keys)
        if any(value is None for value in stratum):
            continue
        groups[stratum][period].append(row)

    capacities = {
        stratum: min(len(by_period.get(period, ())) for period in periods)
        for stratum, by_period in groups.items()
    }
    capacities = {stratum: count for stratum, count in capacities.items() if count}
    total_capacity = sum(capacities.values())
    if target_per_period > total_capacity:
        raise ValueError(
            f"target {target_per_period} exceeds matched capacity {total_capacity}"
        )

    quotas = {
        stratum: (target_per_period * count) // total_capacity
        for stratum, count in capacities.items()
    }
    remainder = target_per_period - sum(quotas.values())
    remainders = sorted(
        (
            stratum
            for stratum in capacities
        ),
        key=lambda stratum: (
            -((target_per_period * capacities[stratum]) % total_capacity),
            tuple(map(str, stratum)),
        ),
    )
    for stratum in remainders[:remainder]:
        quotas[stratum] += 1

    generator = random.Random(seed)
    selected = []
    for stratum in sorted(quotas, key=lambda value: tuple(map(str, value))):
        quota = quotas[stratum]
        if not quota:
            continue
        for period in periods:
            candidates = list(groups[stratum][period])
            generator.shuffle(candidates)
            selected.extend(
                {**row, "_matched_period": period} for row in candidates[:quota]
            )

    counts = {period: 0 for period in periods}
    for row in selected:
        counts[row["_matched_period"]] += 1
    if any(count != target_per_period for count in counts.values()):
        raise RuntimeError(f"internal quota error: {counts}")

    if group_key is not None:
        group_periods = {}
        for row in selected:
            group = row.get(group_key)
            period = row["_matched_period"]
            if group is None:
                raise ValueError(f"selected row is missing group key {group_key!r}")
            previous_period = group_periods.get(group)
            if previous_period is not None:
                raise ValueError(
                    f"group {group!r} appears in periods "
                    f"{previous_period!r} and {period!r}"
                )
            group_periods[group] = period

    return selected
