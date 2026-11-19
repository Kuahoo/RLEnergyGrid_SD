from sdge_microgrid.reward import compute_reward


def test_grid_tied_import_cost_only():
    r, info = compute_reward(
        g=1.0,
        p_crit=2.0,
        p_flex=1.0,
        u_shed=0.0,
        p_pv=1.0,
        p_batt=0.0,
        c_buy=0.691,
        w_flex=1.5,
        w_crit=25.0,
    )
    assert abs(r - (-1.382)) < 1e-9
    assert info["l_crit"] == 0.0
    assert abs(info["c_grid"] - 1.382) < 1e-9


def test_islanded_unmet_critical_penalty():
    r, info = compute_reward(
        g=0.0,
        p_crit=2.0,
        p_flex=1.0,
        u_shed=0.0,
        p_pv=0.0,
        p_batt=0.0,
        c_buy=0.691,
        w_flex=1.5,
        w_crit=25.0,
    )
    assert abs(info["l_crit"] - 2.0) < 1e-9
    assert abs(r - (-50.0)) < 1e-9


def test_flex_shed_penalty_and_degradation():
    r, info = compute_reward(
        g=1.0,
        p_crit=1.0,
        p_flex=2.0,
        u_shed=0.5,
        p_pv=0.0,
        p_batt=4.0,
        c_buy=0.374,
        w_flex=1.5,
        w_crit=25.0,
    )
    assert abs(info["c_deg"] - 0.08) < 1e-9
    assert abs(info["c_flex"] - 1.5) < 1e-9
    assert abs(r - (-1.48)) < 1e-9


def test_islanded_battery_discharge_covers_critical():
    r, info = compute_reward(
        g=0.0,
        p_crit=2.0,
        p_flex=1.0,
        u_shed=0.0,
        p_pv=1.0,
        p_batt=1.5,
        c_buy=0.691,
        w_flex=1.5,
        w_crit=25.0,
    )
    assert info["l_crit"] == 0.0
    assert abs(r - (-(0.02 * 1.5))) < 1e-9
