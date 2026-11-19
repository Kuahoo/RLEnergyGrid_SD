import gymnasium as gym
import sdge_microgrid.env.register  # noqa: F401

from sdge_microgrid.baselines.economic import economic_action
from sdge_microgrid.constants import ENV_ID
from sdge_microgrid.eval.runner import run_episode


def test_run_episode_economic(processed_data_dir):
    env = gym.make(ENV_ID, data_dir=str(processed_data_dir), start_hour=0, scenario="normal")
    result = run_episode(env, economic_action, seed=0)
    assert len(result.infos) == 168
    assert isinstance(result.episode_return, float)
    assert 0.0 <= result.critical_survival_rate <= 100.0
    assert result.flexible_curtailment_ratio == 0.0
    env.close()


def test_run_episode_highrisk_uses_options(processed_data_dir):
    env = gym.make(ENV_ID, data_dir=str(processed_data_dir))
    result = run_episode(
        env,
        economic_action,
        seed=1,
        options={"start_hour": 2000, "scenario": "highrisk"},
    )
    assert len(result.infos) == 168
    assert any(info["g"] == 0.0 for info in result.infos)
    env.close()
