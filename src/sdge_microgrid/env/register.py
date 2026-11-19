from gymnasium.envs.registration import register

from sdge_microgrid.constants import ENV_ID

register(
    id=ENV_ID,
    entry_point="sdge_microgrid.env.microgrid_env:SDGEMicrogridEnv",
)
