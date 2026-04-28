"""
Liquid manipulation task, Direct implementation
"""


import gymnasium as gym

from . import agents
from .psm_head_blood_pipe_env import PsmBloodPipeAbsorptionEnv, PsmBloodPipeAbsorptionEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Psm-Blood-Pipe-State-Direct-v0",
    entry_point="head_blood_absorption.tasks.blood_pipe_state:PsmBloodPipeAbsorptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PsmBloodPipeAbsorptionEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PsmBloodAbsorptionPPORunnerCfg",
    },
)
