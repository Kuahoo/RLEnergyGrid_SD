from __future__ import annotations

import numpy as np


def make_sb3_policy(model):
    """Adapt an SB3 model to the ``policy(obs) -> action`` runner interface."""

    def _policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(2)

    return _policy
