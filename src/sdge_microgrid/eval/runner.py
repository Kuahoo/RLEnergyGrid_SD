from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from sdge_microgrid.eval.metrics import (
    critical_survival_rate,
    flexible_curtailment_ratio,
    weekly_net_economic_cost,
)

Policy = Callable[[np.ndarray], np.ndarray]


@dataclass
class EpisodeResult:
    episode_return: float
    infos: list[dict] = field(default_factory=list)
    critical_survival_rate: float = 100.0
    weekly_net_economic_cost: float = 0.0
    flexible_curtailment_ratio: float = 0.0


def run_episode(
    env,
    policy: Policy,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> EpisodeResult:
    obs, _ = env.reset(seed=seed, options=options)
    infos: list[dict] = []
    total = 0.0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
        total += float(reward)
        if terminated or truncated:
            break
    return EpisodeResult(
        episode_return=total,
        infos=infos,
        critical_survival_rate=critical_survival_rate(infos),
        weekly_net_economic_cost=weekly_net_economic_cost(infos),
        flexible_curtailment_ratio=flexible_curtailment_ratio(infos),
    )
