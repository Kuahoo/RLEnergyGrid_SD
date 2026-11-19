import gymnasium as gym
import numpy as np
import sdge_microgrid.env.register  # noqa: F401

from sdge_microgrid.constants import ENV_ID, EPISODE_HOURS, OBS_DIM


def test_env_spaces_and_episode(processed_data_dir):
    env = gym.make(ENV_ID, data_dir=str(processed_data_dir), start_hour=0, scenario="normal")
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.action_space.shape == (2,)
    total_r = 0.0
    for _ in range(EPISODE_HOURS):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        assert obs.shape == (OBS_DIM,)
        assert 0.1 - 1e-6 <= obs[0] <= 0.95 + 1e-6
        if trunc or term:
            break
    assert trunc is True
    assert info["t"] == EPISODE_HOURS
    assert np.isfinite(total_r)
    env.close()


def test_island_forces_zero_grid_cost(processed_data_dir):
    env = gym.make(ENV_ID, data_dir=str(processed_data_dir), start_hour=0, scenario="highrisk")
    env.reset(seed=1)
    env.unwrapped.g = 0
    obs, r, term, trunc, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert info["c_grid"] == 0.0
    assert info["p_batt"] >= 0.0
    env.close()


def test_reset_options_override(processed_data_dir):
    env = gym.make(ENV_ID, data_dir=str(processed_data_dir))
    _, info = env.reset(seed=2, options={"start_hour": 2000, "scenario": "highrisk"})
    assert info["t"] == 0
    assert env.unwrapped._week.start_hour == 2000
    assert env.unwrapped._week.rho.mean() > 0.4
    env.close()
