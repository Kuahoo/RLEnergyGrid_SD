from sdge_microgrid.physics import clip_battery_power, eta_factor, next_soc


def test_eta_factor_sign_convention():
    assert eta_factor(5.0) == 0.95
    assert abs(eta_factor(-5.0) - (1.0 / 0.95)) < 1e-12
    assert eta_factor(0.0) == 1.0


def test_next_soc_discharge_and_charge():
    soc1 = next_soc(0.8, 10.0)
    assert abs(soc1 - (0.8 - (10.0 * 1.0 / 27.0) * 0.95)) < 1e-9
    soc2 = next_soc(0.5, -10.0)
    assert abs(soc2 - (0.5 - (-10.0 * 1.0 / 27.0) * (1.0 / 0.95))) < 1e-9


def test_clip_respects_power_limit():
    assert clip_battery_power(0.5, 50.0) <= 10.0
    assert clip_battery_power(0.5, -50.0) >= -10.0


def test_clip_respects_soc_bounds():
    p = clip_battery_power(0.11, 10.0)
    assert next_soc(0.11, p) >= 0.1 - 1e-9
    p2 = clip_battery_power(0.94, -10.0)
    assert next_soc(0.94, p2) <= 0.95 + 1e-9
