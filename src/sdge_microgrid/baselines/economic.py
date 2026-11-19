from __future__ import annotations

import numpy as np

from sdge_microgrid.constants import C_BUY_ON_PEAK


def _hour_from_obs(sin_h: float, cos_h: float) -> int:
    angle = float(np.arctan2(sin_h, cos_h)) % (2.0 * np.pi)
    return int(round(angle * 24.0 / (2.0 * np.pi))) % 24


def economic_action(obs: np.ndarray) -> np.ndarray:
    """Risk-blind TOU arbitrage: discharge on-peak, charge on surplus solar/night."""
    soc, pcrit, pflex, ppv, cbuy, g, rho, sin_h, cos_h = [float(x) for x in obs]
    hour = _hour_from_obs(sin_h, cos_h)
    if cbuy >= C_BUY_ON_PEAK - 1e-6:
        u_batt = 1.0
    elif ppv > pcrit + pflex:
        u_batt = -1.0
    elif hour <= 5:
        u_batt = -0.5
    else:
        u_batt = 0.0
    return np.array([u_batt, 0.0], dtype=np.float32)
