"""
Liquid manipulation task, Direct implementation
"""


import gymnasium as gym

from . import agents
from .psm_blood_absorption_env import PsmBloodAbsorptionEnv, PsmBloodAbsorptionEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Psm-Blood-Vision-Direct-v0",
    entry_point="blood_absorption.tasks.blood_vision:PsmBloodAbsorptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PsmBloodAbsorptionEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PsmBloodAbsorptionPPORunnerCfg",
    },
)
