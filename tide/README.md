# CSU TIDE GPU training

See [`docs/context/TIDE.md`](../docs/context/TIDE.md).

## Interactive (JupyterHub)

1. Start **Jupyter Stack PyTorch Notebook** (`pytorch-notebook`).
2. Request the default shape: **8 CPU, 32 GB RAM, 1x NVIDIA L40** (not A100).
3. Clone/open this repo and run `bash tide/jupyter_bootstrap.sh`.
4. Smoke: `SDGE_REQUIRE_CUDA=1 PYTHONPATH=src python tide/check_gpu.py`
5. Train one seed: `bash tide/launch_ppo_seed.sh 0` or `bash tide/launch_sac_seed.sh 0`.

## Batch jobs

Submit **one container job per (algo, seed)** for longer runs and parallelization
(10 jobs total). Each job should use the `pytorch-notebook` image, 1x L40, and
`tide/launch_{ppo,sac}_seed.sh <seed>`.

## Multi-agent split

- Lane C owns PPO logs under `results/ppo/` and `results/logs/ppo_*.log`
- Lane D owns SAC logs under `results/sac/` and `results/logs/sac_*.log`
- Do not share a single GPU between Lane C and Lane D jobs; use separate jobs/nodes.
