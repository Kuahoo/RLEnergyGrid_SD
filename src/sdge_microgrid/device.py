from __future__ import annotations

import os

import torch


def resolve_device(requested: str | None = None) -> str:
    """Resolve ``cuda``/``cpu``, falling back to CPU when CUDA is unavailable."""
    req = (requested or os.environ.get("SDGE_DEVICE", "cuda")).lower()
    if req == "cpu":
        return "cpu"
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def require_cuda_for_tide() -> str:
    """Hard-fail on TIDE training jobs when ``SDGE_REQUIRE_CUDA=1`` and CUDA is missing."""
    if os.environ.get("SDGE_REQUIRE_CUDA", "0") != "1":
        return resolve_device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required on TIDE (SDGE_REQUIRE_CUDA=1) but torch.cuda.is_available() is False. "
            "Start a pytorch-notebook session with 1x L40 (see docs/context/TIDE.md)."
        )
    return "cuda"


def cuda_summary() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current": idx,
        "name": torch.cuda.get_device_name(idx),
        "capability": torch.cuda.get_device_capability(idx),
    }
