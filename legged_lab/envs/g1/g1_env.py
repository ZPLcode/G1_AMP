# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.


import numpy as np
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate

from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


class TienKungG1Env(TienKungEnv):

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"],
            self.num_envs,
            device=self.device,
        )
        self.action_buffer.compute(
            torch.zeros(
                self.num_envs,
                self.num_actions,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(
                time_lags, torch.arange(self.num_envs, device=self.device)
            )

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor",
            body_names=self.cfg.robot.terminate_contacts_body_names,
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.feet_body_names
        )
        self.feet_cfg.resolve(self.scene)

        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        self.wrist_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            preserve_order=True,
        )

        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_hip_yaw_joint",
                "left_hip_roll_joint",
                "left_hip_pitch_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_hip_yaw_joint",
                "right_hip_roll_joint",
                "right_hip_pitch_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.waist_joint_ids, _ = self.robot.find_joints(
            name_keys=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            preserve_order=True,
        )

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )


        self.gait_phase = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.gait_cycle = torch.full(
            (self.num_envs,),
            self.cfg.gait.gait_cycle,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # AMP obs = 23 dof_pos + 23 dof_vel + 12 end_pos + 6 base_vel = 64
        self._amp_obs_dim = (
            2 * (len(self.left_arm_ids) + len(self.right_arm_ids)
                 + len(self.left_leg_ids) + len(self.right_leg_ids)
                 + len(self.waist_joint_ids))
            + 12
            + 6   # root_lin_vel (3) + root_ang_vel (3) in body frame
        )
        self.terminal_amp_obs = torch.zeros(
            self.num_envs, self._amp_obs_dim, dtype=torch.float, device=self.device
        )

        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)

        self.action = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.init_obs_buffer()

        self._init_rsi()

    # ================================================================
    # ================================================================
    def get_amp_obs_for_expert_trans(self):
        body_state = self.robot.data.body_state_w
        root_pos = self.robot.data.root_state_w[:, 0:3]
        root_quat_inv = quat_conjugate(self.robot.data.root_state_w[:, 3:7])

        def _pos_in_root(body_idx):
            rel_in_world = body_state[:, body_idx, :3] - root_pos
            return quat_apply(root_quat_inv, rel_in_world)

        left_hand_pos = _pos_in_root(self.wrist_body_ids[0])
        right_hand_pos = _pos_in_root(self.wrist_body_ids[1])
        left_foot_pos = _pos_in_root(self.feet_body_ids[0])
        right_foot_pos = _pos_in_root(self.feet_body_ids[1])

        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        self.left_leg_dof_pos = joint_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = joint_pos[:, self.right_leg_ids]
        self.waist_dof_pos = joint_pos[:, self.waist_joint_ids]
        self.left_arm_dof_pos = joint_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = joint_pos[:, self.right_arm_ids]
        self.left_leg_dof_vel = joint_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = joint_vel[:, self.right_leg_ids]
        self.waist_dof_vel = joint_vel[:, self.waist_joint_ids]
        self.left_arm_dof_vel = joint_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = joint_vel[:, self.right_arm_ids]

        return torch.cat(
            (
                self.left_leg_dof_pos,    # [0:6]
                self.right_leg_dof_pos,   # [6:12]
                self.waist_dof_pos,       # [12:15]
                self.left_arm_dof_pos,    # [15:19]
                self.right_arm_dof_pos,   # [19:23]
                self.left_leg_dof_vel,    # [23:29]
                self.right_leg_dof_vel,   # [29:35]
                self.waist_dof_vel,       # [35:38]
                self.left_arm_dof_vel,    # [38:42]
                self.right_arm_dof_vel,   # [42:46]
                left_hand_pos,            # [46:49]
                right_hand_pos,           # [49:52]
                left_foot_pos,            # [52:55]
                right_foot_pos,           # [55:58]
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
            ),
            dim=-1,
        )

    # ================================================================
    # RSI (Reference State Initialization)
    # ================================================================
    def _init_rsi(self):
        import json
        import os

        self._rsi_enabled = False
        if not getattr(self.cfg, "rsi_enable", False):
            return
        rsi_file = self.cfg.rsi_data_file
        if not os.path.exists(rsi_file):
            print(f"[WARN] RSI data file not found: {rsi_file}, RSI disabled")
            return

        with open(rsi_file) as f:
            data = json.load(f)
        if "MujocoJointNames" not in data:
            print(f"[WARN] {rsi_file} 不含 MujocoJointNames 字段（不是新格式），RSI disabled")
            return

        frames = np.array(data["Frames"], dtype=np.float32)  # (N, 129)
        N = len(frames)

        # [0:64]    amp_obs
        # [64:67]   root_pos (world)
        # [67:71]   root_quat (wxyz)
        # [71:100]  dof_pos_full (mujoco order, 29)
        # [100:129] dof_vel_full (mujoco order, 29)
        root_pos = frames[:, 64:67]
        root_quat = frames[:, 67:71]
        dof_pos_mj = frames[:, 71:100]
        dof_vel_mj = frames[:, 100:129]

        self._rsi_root_pos = torch.tensor(root_pos, device=self.device)
        self._rsi_root_quat = torch.tensor(root_quat, device=self.device)
        self._rsi_root_lin_vel_b = torch.tensor(frames[:, 58:61], device=self.device)
        self._rsi_root_ang_vel_b = torch.tensor(frames[:, 61:64], device=self.device)

        mujoco_names = list(data["MujocoJointNames"])
        lab_names = list(self.robot.data.joint_names)
        missing = [n for n in lab_names if n not in mujoco_names]
        if missing:
            print(f"[WARN] IsaacLab 关节 {missing} 在 RSI 数据的 MujocoJointNames 里找不到，RSI disabled")
            return
        mj_name_to_idx = {n: i for i, n in enumerate(mujoco_names)}
        mj2lab = torch.tensor(
            [mj_name_to_idx[n] for n in lab_names], device=self.device, dtype=torch.long
        )
        self._rsi_dof_pos = torch.tensor(dof_pos_mj, device=self.device)[:, mj2lab]
        self._rsi_dof_vel = torch.tensor(dof_vel_mj, device=self.device)[:, mj2lab]

        self._rsi_n_frames = N
        self._rsi_prob = float(self.cfg.rsi_prob)
        self._rsi_enabled = True
        print(f"[INFO] ✅ RSI enabled: {N} reference frames, prob={self._rsi_prob}")

    def reset(self, env_ids):
        super().reset(env_ids)

        if len(env_ids) == 0 or not getattr(self, "_rsi_enabled", False):
            return

        n = len(env_ids)
        use_rsi = torch.rand(n, device=self.device) < self._rsi_prob
        rsi_env_ids = env_ids[use_rsi]
        if len(rsi_env_ids) == 0:
            return

        n_rsi = len(rsi_env_ids)
        frame_idx = torch.randint(0, self._rsi_n_frames, (n_rsi,), device=self.device)

        root_quat = self._rsi_root_quat[frame_idx]   # (n_rsi, 4) wxyz
        root_state = torch.zeros((n_rsi, 13), device=self.device, dtype=torch.float)
        root_state[:, 0:2] = self.scene.env_origins[rsi_env_ids, 0:2]
        root_state[:, 2] = self._rsi_root_pos[frame_idx, 2]
        root_state[:, 3:7] = root_quat
        # body frame vel → world frame vel
        lin_vel_b = self._rsi_root_lin_vel_b[frame_idx]
        ang_vel_b = self._rsi_root_ang_vel_b[frame_idx]
        root_state[:, 7:10] = quat_apply(root_quat, lin_vel_b)
        root_state[:, 10:13] = quat_apply(root_quat, ang_vel_b)
        self.robot.write_root_state_to_sim(root_state, rsi_env_ids)

        dof_pos = self._rsi_dof_pos[frame_idx]
        dof_vel = self._rsi_dof_vel[frame_idx]
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=rsi_env_ids)
