from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("results/tables/comparison.csv")
    if not path.exists():
        print(f"No comparison table at {path}; run scripts/evaluate_all.py first.")
        return
    df = pd.read_csv(path)
    high = df[df["scenario"] == "highrisk"]
    normal = df[df["scenario"] == "normal"]
    cons_cost = normal[normal["controller"] == "conservative"][
        "weekly_net_economic_cost"
    ].mean()
    for ctrl in ("ppo", "sac"):
        if ctrl not in set(df["controller"]):
            print(f"{ctrl}: no rows yet")
            continue
        surv = high[high["controller"] == ctrl]["critical_survival_rate"].mean()
        rl_cost = normal[normal["controller"] == ctrl]["weekly_net_economic_cost"].mean()
        print(f"{ctrl} highrisk survival mean={surv:.2f}% (target >98)")
        if cons_cost:
            reduction = 100.0 * (cons_cost - rl_cost) / abs(cons_cost)
            print(
                f"{ctrl} normal-week cost reduction vs conservative="
                f"{reduction:.1f}% (target >15)"
            )


if __name__ == "__main__":
    main()
