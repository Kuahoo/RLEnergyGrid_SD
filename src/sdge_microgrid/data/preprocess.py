"""Optional parsers for NREL PVWatts / OpenEI TMY3 CSVs (place files in data/raw/).

Not used by the default synthetic pipeline; provided so real San Diego traces can
be swapped in without touching the env or training code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sdge_microgrid.constants import P_CRIT_MAX, P_FLEX_MAX, P_PV_MAX


def _scale(values: np.ndarray, ceiling: float) -> np.ndarray:
    peak = float(values.max()) if values.size else 0.0
    return values * (ceiling / peak) if peak > ceiling else values


def preprocess_openei_residential(
    csv_path: Path | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split OpenEI residential end-use columns into critical and flexible load (kW)."""
    df = pd.read_csv(csv_path)
    crit_cols = [c for c in df.columns if "Interior" in c or "Lighting" in c]
    flex_cols = [
        c for c in df.columns if any(k in c for k in ("Heating", "Cooling", "Water"))
    ]
    if not crit_cols or not flex_cols:
        raise ValueError(f"Unexpected OpenEI columns: {list(df.columns)}")
    pcrit = df[crit_cols].sum(axis=1).to_numpy(dtype=np.float64)
    pflex = df[flex_cols].sum(axis=1).to_numpy(dtype=np.float64)
    return _scale(pcrit, P_CRIT_MAX), _scale(pflex, P_FLEX_MAX)


def preprocess_pvwatts(csv_path: Path | str) -> np.ndarray:
    """Extract hourly AC PV output (kW) from a PVWatts CSV."""
    df = pd.read_csv(csv_path)
    col = "AC System Output (W)" if "AC System Output (W)" in df.columns else df.columns[-1]
    pv_kw = df[col].to_numpy(dtype=np.float64) / 1000.0
    return _scale(pv_kw, P_PV_MAX)
