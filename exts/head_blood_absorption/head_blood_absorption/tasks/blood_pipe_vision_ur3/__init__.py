"""
Liquid manipulation task, Direct implementation
"""


import gymnasium as gym

from . import agents
from .ur3_head_blood_pipe_env import Ur3BloodPipeVisionEnv, Ur3BloodPipeVisionEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ur3-Blood-Pipe-Vision-Direct-v0",
    entry_point="head_blood_absorption.tasks.blood_pipe_vision_ur3:Ur3BloodPipeVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Ur3BloodPipeVisionEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
