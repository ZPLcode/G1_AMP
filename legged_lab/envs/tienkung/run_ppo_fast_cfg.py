# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
快跑实验配置：纯 PPO（无 AMP）+ 高速指令 + 快步频步态时钟

设计动机：
    原版 run 任务受 AMP 判别器约束（专家 run.txt 实际只跑 0.9 m/s），
    策略不敢跑快。本实验关闭 AMP，让 PPO 自由探索高速运动。

主要修改 vs run_cfg.py:
    1. 关闭 AMP，改用纯 PPO（runner_class=OnPolicyRunner, algorithm=PPO）
    2. 速度指令上限 1.0 → 3.0 m/s（最快目标 3 m/s）
    3. 步态周期 0.5s → 0.4s（步频加快 25%）
    4. 腾空相位适度调整以匹配更快步频

预期：
    - reward 可能比 run baseline 低（任务变难）
    - 实际速度应突破 2 m/s（不再受 AMP 限制）
    - 步态可能不如 AMP 自然，但能真正"跑"起来
"""

import math

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from legged_lab.envs.base.base_config import CommandRangesCfg, CommandsCfg
from legged_lab.envs.tienkung.run_cfg import GaitCfg, TienKungRunFlatEnvCfg


@configclass
class FastRunGaitCfg(GaitCfg):
    """快跑步态时钟参数：更短周期 + 略增腾空比例"""

    gait_air_ratio_l: float = 0.55  # 原 run: 0.6
    gait_air_ratio_r: float = 0.55  # 原 run: 0.6
    gait_phase_offset_l: float = 0.55  # 原 run: 0.6
    gait_phase_offset_r: float = 0.05  # 原 run: 0.1
    gait_cycle: float = 0.4  # 原 run: 0.5（步频提升 25%）


@configclass
class TienKungRunPpoFastFlatEnvCfg(TienKungRunFlatEnvCfg):
    """快跑环境配置：继承 run，覆盖步态时钟和速度指令"""

    gait: FastRunGaitCfg = FastRunGaitCfg()
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-1.0, 3.0),  # 关键修改：从 (-0.6, 1.0) 拉宽到 (-1.0, 3.0)
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class TienKungRunPpoFastAgentCfg(RslRlOnPolicyRunnerCfg):
    """快跑 Agent 配置：纯 PPO，关闭 AMP"""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",  # 纯 PPO，不用 AMPPPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner"  # 纯 PPO Runner，不用 AmpOnPolicyRunner
    experiment_name = "run_ppo_fast"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "run_ppo_fast"
    wandb_project = "run_ppo_fast"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
