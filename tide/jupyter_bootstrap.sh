#!/usr/bin/env bash
# Run inside TIDE JupyterHub pytorch-notebook (1x L40, 8 CPU, 32GB).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -U pip
python -m pip install -e ".[dev]"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY
PYTHONPATH=src python scripts/generate_synthetic_data.py
