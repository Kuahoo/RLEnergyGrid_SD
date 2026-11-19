from sdge_microgrid.eval.metrics import (
    critical_survival_rate,
    flexible_curtailment_ratio,
    weekly_net_economic_cost,
)


def test_metrics_on_hand_infos():
    infos = [
        {
            "g": 0.0,
            "l_crit": 0.5,
            "p_crit": 2.0,
            "c_grid": 0.0,
            "c_deg": 0.1,
            "u_shed": 0.5,
            "p_flex": 2.0,
        },
        {
            "g": 1.0,
            "l_crit": 0.0,
            "p_crit": 2.0,
            "c_grid": 1.0,
            "c_deg": 0.0,
            "u_shed": 0.0,
            "p_flex": 2.0,
        },
    ]
    assert abs(critical_survival_rate(infos) - 75.0) < 1e-9
    assert abs(weekly_net_economic_cost(infos) - 1.1) < 1e-9
    assert abs(flexible_curtailment_ratio(infos) - 25.0) < 1e-9


def test_survival_is_100_without_outage():
    infos = [{"g": 1.0, "l_crit": 0.0, "p_crit": 2.0}]
    assert critical_survival_rate(infos) == 100.0


def test_curtailment_zero_when_no_flex_load():
    infos = [{"u_shed": 1.0, "p_flex": 0.0}]
    assert flexible_curtailment_ratio(infos) == 0.0
