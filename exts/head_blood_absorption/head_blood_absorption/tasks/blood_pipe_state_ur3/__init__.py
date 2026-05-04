"""
Liquid manipulation task, Direct implementation
"""


import gymnasium as gym

from . import agents
from .ur3_head_blood_pipe_env import Ur3BloodPipeAbsorptionEnv, Ur3BloodPipeAbsorptionEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ur3-Blood-Pipe-State-Direct-v0",
    entry_point="head_blood_absorption.tasks.blood_pipe_state_ur3:Ur3BloodPipeAbsorptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Ur3BloodPipeAbsorptionEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3BloodPipeAbsorptionPPORunnerCfg",
        "dreamer_cfg_entry_point": f"{agents.__name__}:dreamer_cfg.yaml",
        "r2dreamer_cfg_entry_point": f"{agents.__name__}:r2dreamer_cfg.yaml",
    },
)
