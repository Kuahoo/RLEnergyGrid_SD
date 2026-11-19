import numpy as np

from sdge_microgrid.eval.sb3_policy import make_sb3_policy


class _Dummy:
    def predict(self, obs, deterministic=True):
        return np.array([0.0, 0.0], dtype=np.float32), None


def test_adapter_returns_action_vector():
    policy = make_sb3_policy(_Dummy())
    a = policy(np.zeros(9, dtype=np.float32))
    assert a.shape == (2,)
