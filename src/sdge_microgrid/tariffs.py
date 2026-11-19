from __future__ import annotations

from sdge_microgrid.constants import (
    C_BUY_ON_PEAK,
    C_BUY_SUPER_OFF,
    C_SELL,
    DT,
    ON_PEAK_HOURS,
)


def buy_price(hour: int) -> float:
    return C_BUY_ON_PEAK if int(hour) % 24 in ON_PEAK_HOURS else C_BUY_SUPER_OFF


def grid_cost(p_net: float, c_buy: float, dt: float = DT) -> float:
    """C_grid = c_buy * max(P_net,0)*dt - c_sell * max(-P_net,0)*dt."""
    return c_buy * max(p_net, 0.0) * dt - C_SELL * max(-p_net, 0.0) * dt
