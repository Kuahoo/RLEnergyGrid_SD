# SDG&E Microgrid PSPS RL

Risk-aware solar-plus-storage dispatch under SDG&E TOU-DR1 tariffs and stochastic
PSPS outages. Gymnasium environment id: **`SDGEMicrogridPSPS-v0`**.

Hourly MDP: battery (27 kWh, 10 kW, eta 0.95), TOU-DR1 buy prices, a rho-driven
two-state grid Markov chain, and a reward penalizing grid cost, degradation,
flexible-load shedding, and unmet critical load. PPO and SAC (Stable-Baselines3)
are compared against economic and conservative rule baselines on critical
survival, weekly net cost, and flexible curtailment.

Design spec: [`docs/superpowers/specs/2026-11-12-sdge-microgrid-psps-design.md`](docs/superpowers/specs/2026-11-12-sdge-microgrid-psps-design.md)
Implementation plan: [`docs/superpowers/plans/2026-11-12-sdge-microgrid-psps-rl.md`](docs/superpowers/plans/2026-11-12-sdge-microgrid-psps-rl.md)

## Install

```bash
python -m pip install -e ".[dev]"
```

## Generate data

```bash
PYTHONPATH=src python scripts/generate_synthetic_data.py
```

Writes deterministic synthetic San Diego traces to `data/processed/`
(`pv.npy`, `pcrit.npy`, `pflex.npy`, `rho_normal.npy`, `rho_highrisk.npy`).

## Test

```bash
pytest
```

CPU-only; training smoke tests fall back to CPU when CUDA is unavailable.

## TIDE GPU training

Full training runs on **CSU TIDE** NVIDIA **L40** GPUs with `device: cuda`:
`pytorch-notebook` image, default shape **8 CPU / 32 GB RAM / 1x L40**. Do not
request the scarce A100 (GPU-Advanced) unless justified.
See [`docs/context/TIDE.md`](docs/context/TIDE.md) and [`tide/README.md`](tide/README.md).

```bash
bash tide/jupyter_bootstrap.sh          # inside the notebook: install + generate data
SDGE_REQUIRE_CUDA=1 PYTHONPATH=src python tide/check_gpu.py
bash tide/launch_ppo_seed.sh 0          # one seed, one L40
bash tide/launch_sac_seed.sh 0
bash tide/launch_all_seeds.sh           # convenience fan-out (prefer 10 batch jobs)
```

The protocol is 1e6 steps x 5 seeds for each algorithm (10 runs). Preferred on
TIDE: **one container job per `(algo, seed)`**, each with 1x L40.

## Train locally (CPU smoke)

```bash
PYTHONPATH=src python scripts/train_ppo.py --seed 0 --timesteps 2048 --device cpu
PYTHONPATH=src python scripts/train_sac.py --seed 0 --timesteps 3000 --device cpu
```

`--config-base` selects the base YAML and `--config` an optional overlay
(`configs/tide_l40.yaml`). Models land in `results/{ppo,sac}/seed_{k}/`
(`best_model.zip` from `EvalCallback`, `final_model.zip`, `evaluations.npz`).

### Multi-agent lane ownership

| Lane | Owns |
|------|------|
| A Env Core | `constants`, `tariffs`, `physics`, `data/`, `env/`, `baselines/` |
| B Risk | `reward`, `outage`, reward sensitivity |
| C PPO | `results/ppo/`, PPO training runs |
| D SAC | `results/sac/`, SAC training runs |
| E Eval/Plots | `eval/`, evaluation, figures, hypotheses |

## Evaluate and plot

```bash
PYTHONPATH=src python scripts/evaluate_all.py --seeds 0,1,2,3,4
PYTHONPATH=src python scripts/plot_results.py
PYTHONPATH=src python scripts/reward_sensitivity.py
PYTHONPATH=src python scripts/check_hypotheses.py
```

Outputs:

- `results/tables/comparison.csv` — per-controller, per-seed, per-scenario metrics
- `results/tables/reward_sensitivity.csv` — `w_crit x w_flex` sweep
- `results/figures/learning_curves_ppo_sac.png` — mean +/- 1 std across seeds
- `results/figures/trajectory_72h.png` — SoC, PV, shedding, grid availability

Hypothesis checks (proposal section 4): RL critical survival during outages > 98%,
RL normal-week bills > 15% lower than the conservative baseline.

> The artifacts currently under `results/` are **CPU smoke output** (seed 0,
> 2,560 PPO / 3,000 SAC steps), not the 1e6-step x 5-seed protocol. They prove
> the pipeline end to end; their hypothesis numbers are not experimental
> findings. Regenerate on TIDE, then rerun the evaluation commands above.

## Optional real data

Place NREL PVWatts and OpenEI TMY3 CSVs in `data/raw/` and use
`sdge_microgrid.data.preprocess` (`preprocess_pvwatts`,
`preprocess_openei_residential`) to build traces. The default pipeline is
synthetic so tests and CI need no network access.

## Backup plan

If continuous control stalls: (1) discretize to six operating modes and train DQN,
or (2) fix grid availability `g_t = 1` and solve grid-tied TOU arbitrage only.

Six backup modes `[u_batt, u_shed]`:

| # | Mode | Action |
|---|------|--------|
| 1 | Idle | `[0, 0]` |
| 2 | Charge full | `[-1, 0]` |
| 3 | Discharge full | `[1, 0]` |
| 4 | Discharge + shed | `[1, 0.5]` |
| 5 | Reserve + shed | `[0, 1]` |
| 6 | Charge + shed | `[-0.5, 0.5]` |
