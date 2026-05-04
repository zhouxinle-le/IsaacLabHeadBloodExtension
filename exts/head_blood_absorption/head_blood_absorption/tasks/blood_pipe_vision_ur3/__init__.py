"""
Liquid manipulation task, Direct implementation
"""


import gymnasium as gym

from . import agents
from .ur3_head_blood_pipe_env import Ur3BloodPipeVisionEnv, Ur3BloodPipeVisionEnvCfg
from .ur3_head_blood_pipe_env_wrist import Ur3BloodPipeVisionWristEnv, Ur3BloodPipeVisionWristEnvCfg

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
        "dreamer_cfg_entry_point": f"{agents.__name__}:dreamer_cfg.yaml",
        "r2dreamer_cfg_entry_point": f"{agents.__name__}:r2dreamer_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Ur3-Blood-Pipe-Vision-Wrist-Direct-v0",
    entry_point="head_blood_absorption.tasks.blood_pipe_vision_ur3:Ur3BloodPipeVisionWristEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Ur3BloodPipeVisionWristEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_wrist.yaml",
        "dreamer_cfg_entry_point": f"{agents.__name__}:dreamer_cfg_wrist.yaml",
        "r2dreamer_cfg_entry_point": f"{agents.__name__}:r2dreamer_cfg_wrist.yaml",
    },
)
