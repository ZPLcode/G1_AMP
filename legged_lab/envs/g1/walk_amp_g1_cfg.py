# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.

"""
G1 + AMP 走路任务

基于 walk_ppo_g1_cfg.py 的 G1 配置，加上 AMP-PPO 训练：
    - 复用 G1RewardCfg, TienKungWalkPPOG1FlatEnvCfg 的 env 配置
    - Agent 切换到 AmpOnPolicyRunner + AMPPPO
    - 用 g1_lafan_walk.txt 作为专家动捕数据

专家数据格式：
    - 52 dim/帧 (20 dof_pos + 20 dof_vel + 12 end_pos) 与 TienKung 完全兼容
    - 50 FPS, 184 帧 = 3.68 秒走路片段
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

import legged_lab.mdp as mdp

# 复用 walk_ppo_g1 的 env 配置（地形、奖励、域随机化等）
from legged_lab.envs.g1.walk_ppo_g1_cfg import TienKungWalkPPOG1FlatEnvCfg

# G1 资产：用本地 unitree.G1_CFG（含 waist + elbow + wrist，不含手指）
# 比 IsaacLab 自带的 G1_MINIMAL_CFG（含 inspire hand）更适合 AMP，因为关节命名能对上数据
from legged_lab.assets.unitree.unitree import G1_CFG


@configclass
class TienKungWalkAMPG1FlatEnvCfg(TienKungWalkPPOG1FlatEnvCfg):
    """G1 + AMP 训练的 Env 配置，复用 walk_ppo_g1 的所有设置。"""

    # 可视化用的动捕数据（play_amp_animation.py 用）
    amp_motion_files_display = ["legged_lab/envs/g1/datasets/g1_lafan_walk.txt"]

    # ====== Reference State Initialization (RSI) ======
    # reset 时 rsi_prob 概率从 expert 数据抽随机一帧 spawn G1
    # 让 policy 直接从"走路中段"开始练，加速训练 + 防判别器死锁
    # 注意：用同一个 .txt 文件（含 RSI 字段在每帧 [64:129]）
    rsi_enable: bool = True
    rsi_prob: float = 0.85
    rsi_data_file: str = "legged_lab/envs/g1/datasets/g1_lafan_walk.txt"

    def __post_init__(self):
        # 把 walk_ppo_g1 默认的 G1_MINIMAL_CFG（inspire hand 版）
        # 替换成本地 G1_CFG（含 waist/elbow/wrist，关节名与 AMP 数据匹配）
        self.scene.robot = G1_CFG

        # 新 G1 资产的关节命名和 walk_ppo_g1 默认（inspire hand 版）不同：
        #   - 肘:   elbow_pitch_joint → elbow_joint，无 elbow_roll
        #   - 腰:   torso_joint → waist_yaw + waist_roll + waist_pitch（3 个）
        #   - 无手指（fingers reward 要去掉）
        # 全部用 joint_deviation_l1_always（无论是否运动都约束），避免 policy
        # 走路时手臂乱摆 → 判别器靠看幅度作弊。
        # ---- joint_deviation_hip：把 elbow_pitch 改成 elbow ----
        self.reward.joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.15,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_yaw_joint",
                        ".*_hip_roll_joint",
                        ".*_shoulder_pitch_joint",
                        ".*_elbow_joint",
                    ],
                )
            },
        )
        # ---- joint_deviation_arms：去掉 elbow_roll（新资产没有） ----
        # weight 从 -0.2 提到 -0.5（之前训练发现 joint_deviation_arms 累积大 -0.065，手臂偏 default 远）
        self.reward.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                    ],
                )
            },
        )
        # ---- joint_deviation_torso：单 torso → 3 个 waist 关节 ----
        self.reward.joint_deviation_torso = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "waist_yaw_joint",
                        "waist_roll_joint",
                        "waist_pitch_joint",
                    ],
                )
            },
        )
        # ---- 新增：约束 wrist 不要乱晃（新资产有 wrist 关节） ----
        self.reward.joint_deviation_wrist = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*"])},
        )
        # ---- 去掉手指惩罚（新资产没手指） ----
        self.reward.joint_deviation_fingers = None

        # ====== AMP 训练特化（Y 路线：TienKung-style + LAFAN1 节奏对齐） ======

        # 1. 完全去掉 gait 时钟 reward（避免跟 AMP 节奏冲突，让 AMP 主导步频）
        self.reward.gait_feet_frc_perio = None
        self.reward.gait_feet_spd_perio = None
        self.reward.gait_feet_frc_support_perio = None

        # 2. 删除 feet_y_distance：hardcoded 0.299m 是 TienKung 髋宽，G1 ~0.2m 不匹配
        self.reward.feet_y_distance = None

        # 3. action_rate_l2 加回 -0.01（之前减半到 -0.005 导致 action 抖动太大，sim2sim 跟不上）
        self.reward.action_rate_l2.weight = -0.01

        # 2. 加 fly：禁止双脚同时离地（防止 G1 用"小跳"前进）
        self.reward.fly = RewTerm(
            func=mdp.fly,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
                "threshold": 1.0,
            },
        )

        # 3. 加 feet_air_time：奖励单脚支撑 >0.4s（防止原地踏步/拖步）
        self.reward.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=0.15,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
                "threshold": 0.4,
            },
        )

        # 4. arm_joint_vel_l2：惩罚手臂关节速度（防手臂高频抖动）
        #    weight 从 -0.002 提到 -0.01（5x），之前累积 -0.146 太大说明手臂还在抖
        self.reward.arm_joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_shoulder_.*", ".*_elbow.*", ".*_wrist.*"]
                )
            },
        )


@configclass
class TienKungWalkAMPG1AgentCfg(RslRlOnPolicyRunnerCfg):
    """G1 AMP-PPO Agent 配置。"""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = True    # 开启 actor/critic obs running mean/var 归一化，让 PPO 训练更稳

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",  # ← 关键：AMPPPO（不是普通 PPO）
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
    runner_class_name = "AmpOnPolicyRunner"  # ← 关键：AMP Runner
    experiment_name = "walk_amp_g1"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_amp_g1"
    wandb_project = "walk_amp_g1"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # ============================================================
    # AMP 专属参数
    # ============================================================
    # ⬇ 严格对齐 G1DWAQ_Lab 参考实现
    # 注意：amp_motion_weights 在参考 runner 里支持，但本仓库 amp_on_policy_runner.py
    # 只读 amp_motion_files，没有这个字段，加了也不会被读，所以不加。
    amp_reward_coef = 2.0                 # AMP 信号 ×2（让 expert motion 强引导，对抗 sim-specific bias）
    amp_motion_files = [
        "legged_lab/envs/g1/datasets/g1_lafan_walk.txt",
    ]
    amp_num_preload_transitions = 100000
    amp_task_reward_lerp = 0.3            # AMP 70% + task 30%（AMP 主导步态，sim2sim 鲁棒）
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 23      # 动作标准差下界（23 = AMP 关节子集，含 waist）
