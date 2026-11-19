"""Synthetic San Diego year traces respecting the proposal's power bounds.

Real data (NREL PVWatts / OpenEI TMY3) can be substituted via ``preprocess.py``;
CI and the default experiments use these deterministic synthetic traces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sdge_microgrid.constants import P_CRIT_MAX, P_FLEX_MAX, P_PV_MAX

YEAR_HOURS = 8760
# Fire-weather (high-rho) bursts, long enough to cover the held-out eval weeks.
BURST_STARTS = (50, 2000, 4500, 7000)
BURST_HOURS = 120


def generate_year_traces(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hours = np.arange(YEAR_HOURS)
    hod = hours % 24

    # Solar: half-sine daylight shape peaking at noon, day-to-day variability.
    solar = np.clip(np.sin((hod - 6) / 24 * 2 * np.pi) * P_PV_MAX, 0.0, P_PV_MAX)
    solar = solar * (0.7 + 0.3 * rng.random(YEAR_HOURS))

    # Critical load: always-on base plus evening bump.
    pcrit = 1.0 + 0.8 * ((hod >= 18) & (hod <= 22)) + 0.2 * rng.random(YEAR_HOURS)
    pcrit = np.clip(pcrit, 0.0, P_CRIT_MAX)

    # Flexible load: daytime base plus evening peak, fully deferrable.
    pflex = 2.0 + 2.5 * ((hod >= 17) & (hod <= 22)) + rng.random(YEAR_HOURS)
    pflex = np.clip(pflex, 0.0, P_FLEX_MAX)

    rho_normal = rng.uniform(0.0, 0.05, size=YEAR_HOURS)
    rho_highrisk = rho_normal.copy()
    for start in BURST_STARTS:
        rho_highrisk[start : start + BURST_HOURS] = rng.uniform(
            0.4, 0.95, size=BURST_HOURS
        )

    return {
        "pv": solar.astype(np.float64),
        "pcrit": pcrit.astype(np.float64),
        "pflex": pflex.astype(np.float64),
        "rho_normal": rho_normal.astype(np.float64),
        "rho_highrisk": rho_highrisk.astype(np.float64),
    }


def save_processed(directory: Path | str, traces: dict[str, np.ndarray]) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for key, values in traces.items():
        np.save(d / f"{key}.npy", values)
