from sdge_microgrid.eval.sensitivity import SWEEP


def test_sweep_covers_proposal_defaults():
    assert (25.0, 1.5) in SWEEP
    assert len(SWEEP) == 9
