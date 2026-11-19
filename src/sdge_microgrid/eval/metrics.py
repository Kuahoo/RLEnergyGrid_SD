from __future__ import annotations

from sdge_microgrid.constants import DT


def critical_survival_rate(infos: list[dict]) -> float:
    """Share of critical energy met during islanded hours (%); 100 if never islanded."""
    crit_island = 0.0
    unmet = 0.0
    for info in infos:
        if float(info["g"]) < 0.5:
            crit_island += float(info["p_crit"]) * DT
            unmet += float(info["l_crit"])
    if crit_island <= 0.0:
        return 100.0
    return 100.0 * (crit_island - unmet) / crit_island


def weekly_net_economic_cost(infos: list[dict]) -> float:
    return float(sum(float(i["c_grid"]) + float(i["c_deg"]) for i in infos))


def flexible_curtailment_ratio(infos: list[dict]) -> float:
    """Share of flexible-load energy shed (%)"""
    shed = 0.0
    total = 0.0
    for info in infos:
        energy = float(info["p_flex"]) * DT
        shed += float(info["u_shed"]) * energy
        total += energy
    if total <= 0.0:
        return 0.0
    return 100.0 * shed / total
