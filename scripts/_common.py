"""Shared helpers for the command-line scripts (training, eval, plots)."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sdge_microgrid.env.register  # noqa: F401
from sdge_microgrid.constants import ENV_ID
from sdge_microgrid.device import require_cuda_for_tide, resolve_device

ALGOS = {"ppo": PPO, "sac": SAC}
EVAL_FREQ = 10_000


def load_config(
    path: Path | str = "configs/default.yaml",
    overlay: Path | str | None = None,
) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if overlay is not None:
        over = yaml.safe_load(Path(overlay).read_text()) or {}
        for key, value in over.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value
    return cfg


def make_vec_env(data_dir: str | Path, seed: int) -> DummyVecEnv:
    def _thunk():
        env = Monitor(gym.make(ENV_ID, data_dir=str(data_dir)))
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_thunk])


def model_path(results_dir: Path | str, algo: str, seed: int) -> Path:
    """Prefer the EvalCallback best model, fall back to the final checkpoint."""
    seed_dir = Path(results_dir) / algo / f"seed_{seed}"
    best = seed_dir / "best_model.zip"
    return best if best.exists() else seed_dir / "final_model.zip"


def train_main(algo: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-base", default="configs/default.yaml")
    parser.add_argument("--config", default=None, help="optional overlay e.g. configs/tide_l40.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None, help="cuda|cpu (default from config/env)")
    args = parser.parse_args()

    cfg = load_config(args.config_base, args.config)
    requested = args.device or cfg.get("device", "cuda")
    device = require_cuda_for_tide() if requested == "cuda" else resolve_device(requested)
    timesteps = args.timesteps or int(cfg["train_timesteps"])
    data_dir = args.data_dir or cfg["data_dir"]
    out = Path(cfg["results_dir"]) / algo / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(data_dir, args.seed)
    eval_env = make_vec_env(data_dir, args.seed + 10_000)
    model = ALGOS[algo](
        "MlpPolicy",
        train_env,
        device=device,
        gamma=float(cfg["gamma"]),
        seed=args.seed,
        verbose=1,
        **cfg[algo],
    )
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out),
        log_path=str(out),
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(str(out / "final_model"))
    np.savez(out / "train_meta.npz", seed=args.seed, timesteps=timesteps, device=device)
    print(f"Saved {algo.upper()} seed={args.seed} device={device} to {out}")
