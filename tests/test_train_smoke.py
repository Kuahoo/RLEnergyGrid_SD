from pathlib import Path

import yaml


def _cfg() -> dict:
    return yaml.safe_load(Path("configs/default.yaml").read_text())


def test_default_config_has_ppo_block():
    cfg = _cfg()
    assert cfg["ppo"]["batch_size"] == 64
    assert cfg["train_timesteps"] == 1000000


def test_default_config_has_sac_block():
    cfg = _cfg()
    assert cfg["sac"]["buffer_size"] == 100000
    assert cfg["n_seeds"] == 5
