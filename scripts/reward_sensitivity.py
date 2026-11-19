from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym

import sdge_microgrid.env.register  # noqa: F401
from sdge_microgrid.baselines import conservative_action
from sdge_microgrid.constants import ENV_ID
from sdge_microgrid.eval.runner import run_episode
from sdge_microgrid.eval.sensitivity import SWEEP

from _common import ALGOS, load_config, model_path

from sdge_microgrid.eval.sb3_policy import make_sb3_policy


def _sweep(name, policy, data_dir: str, rows: list[dict]) -> None:
    for w_crit, w_flex in SWEEP:
        env = gym.make(
            ENV_ID, data_dir=data_dir, w_crit=w_crit, w_flex=w_flex
        )
        res = run_episode(
            env,
            policy,
            seed=0,
            options={"start_hour": 2000, "scenario": "highrisk"},
        )
        rows.append(
            {
                "w_crit": w_crit,
                "w_flex": w_flex,
                "controller": name,
                "critical_survival_rate": res.critical_survival_rate,
                "weekly_net_economic_cost": res.weekly_net_economic_cost,
                "flexible_curtailment_ratio": res.flexible_curtailment_ratio,
                "episode_return": res.episode_return,
            }
        )
        env.close()


def main() -> None:
    cfg = load_config()
    rows: list[dict] = []
    _sweep("conservative", conservative_action, cfg["data_dir"], rows)
    ppo_path = model_path(cfg["results_dir"], "ppo", 0)
    if ppo_path.exists():
        policy = make_sb3_policy(ALGOS["ppo"].load(str(ppo_path)))
        _sweep("ppo", policy, cfg["data_dir"], rows)

    out = Path(cfg["results_dir"]) / "tables" / "reward_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
