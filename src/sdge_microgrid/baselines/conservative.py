from __future__ import annotations

import numpy as np

from sdge_microgrid.baselines.economic import economic_action
from sdge_microgrid.constants import RHO_RESERVE_THRESHOLD


def conservative_action(obs: np.ndarray) -> np.ndarray:
    """Hold battery reserve during fire-weather warnings.

    Grid-tied: no TOU discharge, so the reserve survives a possible PSPS.
    Islanded: spend the reserve on critical load and shed all flexible load.
    """
    rho = float(obs[6])
    g = float(obs[5])
    if rho >= RHO_RESERVE_THRESHOLD:
        if g < 0.5:
            return np.array([1.0, 1.0], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)
    return economic_action(obs)
