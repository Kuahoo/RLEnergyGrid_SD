from __future__ import annotations

from pathlib import Path

import gymnasium as gym

import sdge_microgrid.env.register  # noqa: F401
from sdge_microgrid.baselines import conservative_action
from sdge_microgrid.constants import ENV_ID
from sdge_microgrid.eval.plots import (
    plot_learning_curves,
    plot_trajectory_72h,
    summarize_comparison_csv,
)
from sdge_microgrid.eval.runner import run_episode
from sdge_microgrid.eval.sb3_policy import make_sb3_policy

from _common import ALGOS, load_config, model_path


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["results_dir"])
    data_dir = cfg["data_dir"]
    fig_dir = results_dir / "figures"
    plot_learning_curves(results_dir, fig_dir / "learning_curves_ppo_sac.png")

    env = gym.make(ENV_ID, data_dir=data_dir)
    ppo_path = model_path(results_dir, "ppo", 0)
    if ppo_path.exists():
        policy = make_sb3_policy(ALGOS["ppo"].load(str(ppo_path)))
    else:
        policy = conservative_action
    res = run_episode(
        env, policy, seed=0, options={"start_hour": 2000, "scenario": "highrisk"}
    )
    plot_trajectory_72h(
        res.infos[:72], fig_dir / "trajectory_72h.png", title="72h dispatch (high-risk)"
    )
    env.close()

    table_path = results_dir / "tables" / "comparison.csv"
    if table_path.exists():
        print(summarize_comparison_csv(table_path))


if __name__ == "__main__":
    main()
