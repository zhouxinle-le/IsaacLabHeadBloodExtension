from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import Ur3BloodPipeAbsorptionPPORunnerCfg


@configclass
class SafePpoLagrangianCfg:
    enabled: bool = True
    cost_limit: float = 0.02
    lambda_init: float = 0.0
    lambda_lr: float = 0.5
    lambda_max: float = 30.0
    cost_value_loss_coef: float = 1.0
    normalize_cost_advantage: bool = True


@configclass
class SafeRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    safety: SafePpoLagrangianCfg = SafePpoLagrangianCfg()


@configclass
class SafeUr3BloodPipeAbsorptionPPORunnerCfg(Ur3BloodPipeAbsorptionPPORunnerCfg):
    experiment_name = "ur3_blood_pipe_state_safe_ppo_lagrangian"
    env = {"cfg_overrides": {"reward_include_safety_penalties": False}}
    algorithm = SafeRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

