from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gymnasium as gym

import sdge_microgrid.env.register  # noqa: F401
from sdge_microgrid.baselines import conservative_action, economic_action
from sdge_microgrid.constants import ENV_ID
from sdge_microgrid.eval.runner import run_episode
from sdge_microgrid.eval.sb3_policy import make_sb3_policy

from _common import ALGOS, load_config, model_path

# Held-out weeks. High-risk starts lie inside synthetic high-rho bursts.
NORMAL_STARTS = [100, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600]
HIGHRISK_STARTS = [100, 2000, 2040, 2100, 4500, 4540, 4600, 7000, 7040, 7100]
FIELDS = [
    "controller",
    "seed",
    "scenario",
    "start_hour",
    "critical_survival_rate",
    "weekly_net_economic_cost",
    "flexible_curtailment_ratio",
    "episode_return",
]


def eval_controller(name, policy, data_dir, seed, rows) -> None:
    env = gym.make(ENV_ID, data_dir=data_dir)
    for scenario, starts in (("normal", NORMAL_STARTS), ("highrisk", HIGHRISK_STARTS)):
        for start in starts:
            res = run_episode(
                env,
                policy,
                seed=seed * 10_000 + start,
                options={"start_hour": start, "scenario": scenario},
            )
            rows.append(
                {
                    "controller": name,
                    "seed": seed,
                    "scenario": scenario,
                    "start_hour": start,
                    "critical_survival_rate": res.critical_survival_rate,
                    "weekly_net_economic_cost": res.weekly_net_economic_cost,
                    "flexible_curtailment_ratio": res.flexible_curtailment_ratio,
                    "episode_return": res.episode_return,
                }
            )
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data_dir"]
    results_dir = cfg["results_dir"]
    seeds = [int(s) for s in args.seeds.split(",")]

    rows: list[dict] = []
    for seed in seeds:
        eval_controller("economic", economic_action, data_dir, seed, rows)
        eval_controller("conservative", conservative_action, data_dir, seed, rows)
        for algo in ("ppo", "sac"):
            path = model_path(results_dir, algo, seed)
            if not path.exists():
                continue
            model = ALGOS[algo].load(str(path))
            eval_controller(algo, make_sb3_policy(model), data_dir, seed, rows)

    out = Path(results_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
