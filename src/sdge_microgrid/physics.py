from __future__ import annotations

from sdge_microgrid.constants import DT, E_NOM, ETA, P_MAX, SOC_MAX, SOC_MIN


def eta_factor(p_batt: float) -> float:
    """eta ** sign(P_batt), with sign(0) = 0."""
    if p_batt > 0.0:
        return ETA
    if p_batt < 0.0:
        return 1.0 / ETA
    return 1.0


def next_soc(soc: float, p_batt: float, dt: float = DT) -> float:
    return soc - (p_batt * dt / E_NOM) * eta_factor(p_batt)


def clip_battery_power(soc: float, p_batt_cmd: float, dt: float = DT) -> float:
    """Clip commanded P_batt so |P| <= P_MAX and next SoC stays in [SOC_MIN, SOC_MAX]."""
    p = float(max(-P_MAX, min(P_MAX, p_batt_cmd)))
    if p > 0.0:
        # soc - (p*dt/E)*ETA >= SOC_MIN
        p = min(p, max(0.0, (soc - SOC_MIN) * E_NOM / (dt * ETA)))
    elif p < 0.0:
        # soc - (p*dt/E)/ETA <= SOC_MAX
        p = max(p, -max(0.0, (SOC_MAX - soc) * E_NOM * ETA / dt))
    return float(p)
