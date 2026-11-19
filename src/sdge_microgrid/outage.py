from __future__ import annotations

import numpy as np


class PSPSMarkovChain:
    """Two-state grid availability chain, outage risk driven by fire-weather warning rho.

    Calibration: the stationary islanded fraction is ``p01 / (p01 + p_restore)``.
    With ``p_restore = 1/36`` (mean 36 h outage, mid of the proposal's 24-48 h):

    - rho = 0    -> p01 = 5e-4  -> ~1.5% of hours islanded (rare baseline outages)
    - rho = 0.05 -> p01 = 1.15e-3 -> ~4% islanded during normal weeks
    - rho = 0.9  -> p01 = 1.22e-2 -> ~30% islanded (roughly one 36 h PSPS per burst)

    The plan's original ``p_outage_base=0.02`` implied ~42% of *normal* hours
    islanded, which contradicts both the "rare when rho ~ 0" intent and the
    outage test.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        p_outage_base: float = 5e-4,
        p_outage_rho_scale: float = 1.3e-2,
        p_restore: float = 1.0 / 36.0,
    ) -> None:
        self.rng = rng
        self.p_outage_base = p_outage_base
        self.p_outage_rho_scale = p_outage_rho_scale
        self.p_restore = p_restore
        self.g = 1

    def reset(self, g0: int = 1) -> int:
        self.g = int(g0)
        return self.g

    def step(self, rho: float) -> int:
        rho = float(np.clip(rho, 0.0, 1.0))
        u = self.rng.random()
        if self.g == 1:
            p_outage = float(
                np.clip(self.p_outage_base + self.p_outage_rho_scale * rho, 0.0, 1.0)
            )
            if u < p_outage:
                self.g = 0
        elif u < self.p_restore:
            self.g = 1
        return self.g
