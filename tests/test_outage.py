import numpy as np

from sdge_microgrid.outage import PSPSMarkovChain


def test_low_rho_rarely_outages():
    chain = PSPSMarkovChain(np.random.default_rng(0))
    chain.reset(1)
    outages = 0
    for _ in range(10_000):
        outages += int(chain.step(rho=0.0) == 0)
    assert outages < 500


def test_high_rho_can_enter_outage():
    chain = PSPSMarkovChain(np.random.default_rng(1))
    chain.reset(1)
    saw = False
    for _ in range(500):
        if chain.step(rho=1.0) == 0:
            saw = True
            break
    assert saw


def test_mean_island_duration_near_36h():
    chain = PSPSMarkovChain(np.random.default_rng(2), p_restore=1.0 / 36.0)
    durations = []
    for _ in range(200):
        chain.reset(0)
        d = 1
        while chain.step(rho=0.0) == 0 and d < 500:
            d += 1
        durations.append(d)
    assert 25.0 < float(np.mean(durations)) < 50.0


def test_reset_and_step_return_valid_states():
    chain = PSPSMarkovChain(np.random.default_rng(3))
    assert chain.reset(1) == 1
    assert chain.reset(0) == 0
    assert chain.step(rho=0.5) in (0, 1)
