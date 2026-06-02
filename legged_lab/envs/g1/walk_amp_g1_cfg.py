# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

import legged_lab.mdp as mdp

from legged_lab.envs.g1.walk_ppo_g1_cfg import TienKungWalkPPOG1FlatEnvCfg

from legged_lab.assets.unitree.unitree import G1_CFG


@configclass
class TienKungWalkAMPG1FlatEnvCfg(TienKungWalkPPOG1FlatEnvCfg):

    amp_motion_files_display = ["legged_lab/envs/g1/datasets/g1_lafan_walk.txt"]

    # ====== Reference State Initialization (RSI) ======
    rsi_enable: bool = True
    rsi_prob: float = 0.85
    rsi_data_file: str = "legged_lab/envs/g1/datasets/g1_lafan_walk.txt"

    def __post_init__(self):
        self.scene.robot = G1_CFG

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
        self.reward.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
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
        self.reward.joint_deviation_wrist = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*"])},
        )
        self.reward.joint_deviation_fingers = None


        self.reward.gait_feet_frc_perio = None
        self.reward.gait_feet_spd_perio = None
        self.reward.gait_feet_frc_support_perio = None

        self.reward.feet_y_distance = None

        self.reward.action_rate_l2.weight = -0.005

        self.reward.fly = RewTerm(
            func=mdp.fly,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
                "threshold": 1.0,
            },
        )

        self.reward.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=0.15,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
                "threshold": 0.4,
            },
        )

        self.reward.arm_joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.002,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_shoulder_.*", ".*_elbow.*", ".*_wrist.*"]
                )
            },
        )


@configclass
class TienKungWalkAMPG1AgentCfg(RslRlOnPolicyRunnerCfg):

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
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
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "walk_amp_g1"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_amp_g1"
    wandb_project = "walk_amp_g1"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

  
    amp_reward_coef = 1.0
    amp_motion_files = [
        "legged_lab/envs/g1/datasets/g1_lafan_walk.txt",
    ]
    amp_num_preload_transitions = 100000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 23
