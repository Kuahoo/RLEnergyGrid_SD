#!/usr/bin/env bash
# One TIDE batch/interactive job: SAC for a single seed on 1x L40.
set -euo pipefail
SEED="${1:?usage: launch_sac_seed.sh <seed> [gpu_index]}"
GPU_INDEX="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export SDGE_REQUIRE_CUDA=1
export SDGE_DEVICE=cuda
export PYTHONPATH=src
mkdir -p results/logs
python tide/check_gpu.py
python scripts/train_sac.py --config-base configs/default.yaml --config configs/tide_l40.yaml \
  --seed "$SEED" --device cuda
