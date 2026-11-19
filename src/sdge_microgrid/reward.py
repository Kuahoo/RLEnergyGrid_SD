from __future__ import annotations

from sdge_microgrid.constants import C_DEG, DT, W_CRIT, W_FLEX
from sdge_microgrid.tariffs import grid_cost


def compute_reward(
    *,
    g: float,
    p_crit: float,
    p_flex: float,
    u_shed: float,
    p_pv: float,
    p_batt: float,
    c_buy: float,
    w_flex: float = W_FLEX,
    w_crit: float = W_CRIT,
    dt: float = DT,
) -> tuple[float, dict]:
    """Risk-aware reward: negative of (grid cost + degradation + shed + critical shortfall)."""
    u_shed = float(min(1.0, max(0.0, u_shed)))
    p_served = p_crit + (1.0 - u_shed) * p_flex
    p_net = p_served - p_pv - p_batt

    if g >= 0.5:
        c_grid = grid_cost(p_net, c_buy, dt)
        l_crit = 0.0
    else:
        c_grid = 0.0
        p_avail = p_pv + max(p_batt, 0.0)
        l_crit = max(p_crit - p_avail, 0.0) * dt

    c_deg = C_DEG * abs(p_batt) * dt
    c_flex = w_flex * (u_shed * p_flex * dt)
    c_crit = w_crit * l_crit
    reward = -(c_grid + c_deg + c_flex + c_crit)
    info = {
        "p_served": p_served,
        "p_net": p_net,
        "c_grid": c_grid,
        "c_deg": c_deg,
        "c_flex": c_flex,
        "l_crit": l_crit,
        "c_crit": c_crit,
    }
    return float(reward), info
