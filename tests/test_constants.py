from sdge_microgrid import constants as C


def test_core_mdp_constants():
    assert C.DT == 1.0
    assert C.E_NOM == 27.0
    assert C.P_MAX == 10.0
    assert C.ETA == 0.95
    assert C.SOC_MIN == 0.1
    assert C.SOC_MAX == 0.95
    assert C.EPISODE_HOURS == 168
    assert C.GAMMA == 0.99
    assert C.C_BUY_ON_PEAK == 0.691
    assert C.C_BUY_SUPER_OFF == 0.374
    assert C.C_SELL == 0.05
    assert C.C_DEG == 0.02
    assert C.W_FLEX == 1.50
    assert C.W_CRIT == 25.00
    assert C.RHO_RESERVE_THRESHOLD == 0.25
    assert C.OBS_DIM == 9
    assert C.ACTION_LOW == (-1.0, 0.0)
    assert C.ACTION_HIGH == (1.0, 1.0)


def test_on_peak_hours():
    assert C.ON_PEAK_HOURS == frozenset({16, 17, 18, 19, 20})
