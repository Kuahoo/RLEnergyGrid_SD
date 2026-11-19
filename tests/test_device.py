from pathlib import Path

import pytest
import yaml

from sdge_microgrid.device import resolve_device


def _cfg() -> dict:
    return yaml.safe_load(Path("configs/default.yaml").read_text())


def test_tide_config_requests_cuda_and_l40():
    cfg = _cfg()
    assert cfg["device"] == "cuda"
    assert cfg["tide"]["gpu_type"] == "L40"
    assert cfg["tide"]["gpus_per_job"] == 1
    assert cfg["tide"]["avoid_gpu_advanced_a100"] is True


def test_resolve_device_cpu_fallback(monkeypatch):
    import sdge_microgrid.device as D

    monkeypatch.setattr(D.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("SDGE_DEVICE", raising=False)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_when_available(monkeypatch):
    import sdge_microgrid.device as D

    monkeypatch.setattr(D.torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"


def test_require_cuda_for_tide_hard_fails(monkeypatch):
    import sdge_microgrid.device as D

    monkeypatch.setattr(D.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SDGE_REQUIRE_CUDA", "1")
    with pytest.raises(RuntimeError, match="CUDA required on TIDE"):
        D.require_cuda_for_tide()


def test_require_cuda_for_tide_soft_fallback(monkeypatch):
    import sdge_microgrid.device as D

    monkeypatch.setattr(D.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SDGE_REQUIRE_CUDA", "0")
    assert D.require_cuda_for_tide() == "cpu"
