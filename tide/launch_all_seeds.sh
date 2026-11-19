#!/usr/bin/env bash
# Fan-out helper for multi-agent / multi-job TIDE sweeps (10 trainings).
# Preferred: submit each line as its own TIDE batch container job, one L40 per job.
# On a single 4x L40 node this helper runs all 10 concurrently on GPU 0; use it
# only for short smoke sweeps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/logs
for SEED in 0 1 2 3 4; do
  nohup bash tide/launch_ppo_seed.sh "$SEED" 0 >"results/logs/ppo_seed_${SEED}.log" 2>&1 &
  echo "queued PPO seed=$SEED pid=$!"
  nohup bash tide/launch_sac_seed.sh "$SEED" 0 >"results/logs/sac_seed_${SEED}.log" 2>&1 &
  echo "queued SAC seed=$SEED pid=$!"
done
echo "Launched 10 trainings. Prefer one job per seed on separate TIDE batch containers."
wait
