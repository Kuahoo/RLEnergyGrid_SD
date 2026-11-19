import numpy as np

from sdge_microgrid.baselines.conservative import conservative_action
from sdge_microgrid.baselines.economic import economic_action


def _obs(**kw):
    base = dict(soc=0.7, pcrit=1, pflex=2, ppv=0, cbuy=0.691, g=1, rho=0.0, s=0, c=1)
    base.update(kw)
    return np.array(
        [
            base["soc"],
            base["pcrit"],
            base["pflex"],
            base["ppv"],
            base["cbuy"],
            base["g"],
            base["rho"],
            base["s"],
            base["c"],
        ],
        dtype=np.float32,
    )


def test_economic_discharges_on_peak():
    a = economic_action(_obs(cbuy=0.691))
    assert a[0] == 1.0 and a[1] == 0.0


def test_economic_charges_with_surplus_solar():
    a = economic_action(_obs(cbuy=0.374, ppv=8, pcrit=1, pflex=1))
    assert a[0] == -1.0


def test_economic_charges_at_night():
    # midnight: sin=0, cos=1 -> hour 0
    a = economic_action(_obs(cbuy=0.374, ppv=0, pcrit=1, pflex=2, s=0.0, c=1.0))
    assert a[0] == -0.5


def test_conservative_halts_discharge_when_rho_high():
    a = conservative_action(_obs(cbuy=0.691, rho=0.3))
    assert a[0] == 0.0


def test_conservative_sheds_when_islanded():
    a = conservative_action(_obs(g=0.0, rho=0.5, cbuy=0.374))
    assert a[1] == 1.0


def test_conservative_delegates_when_rho_low():
    a = conservative_action(_obs(cbuy=0.691, rho=0.0))
    assert a[0] == 1.0
