# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
TienKungG1Env - 让 TienKung-Lab 框架支持 Unitree G1。

实现思路：
    TienKungEnv.init_buffers() 内部硬编码了 TienKung 的 joint/body 名（如
    "ankle_roll_l_link"、"hip_roll_l_joint" 等），无法直接配置化。

    本类完全重写 init_buffers，把硬编码的 TienKung 名换成 G1 名。其余逻辑
    （计数器、步态时钟、AMP 缓存、观测历史等）保持和父类一致。

为什么独立成文件：
    - 不触碰 tienkung_env.py（零风险破坏现有 TienKung 任务）
    - 明确区分两种机器人的配置
    - 未来加新机器人时，按同样模式新建 xxx_env.py 即可

用法：
    在 envs/__init__.py 注册时用 TienKungG1Env 替换 TienKungEnv
"""

import numpy as np
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate

from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


class TienKungG1Env(TienKungEnv):
    """支持 Unitree G1 的 TienKungEnv 子类。

    仅重写 init_buffers 以使用 G1 命名规则：
      TienKung: hip_roll_{l,r}_joint     →  G1: {left,right}_hip_roll_joint
      TienKung: ankle_roll_{l,r}_link    →  G1: {left,right}_ankle_roll_link
      TienKung: elbow_pitch_{l,r}_link   →  G1: {left,right}_elbow_pitch_link
      TienKung: knee_pitch_{l,r}_joint   →  G1: {left,right}_knee_joint  (注意：G1 无 pitch)
    """

    def init_buffers(self):
        """初始化所有训练过程中需要的 GPU Tensor 和索引映射（G1 版本）。"""
        self.extras = {}

        # ====== A. 基础训练参数 ======
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]  # G1: 通常 23+ DOF
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        # ====== B. 动作延迟缓冲区 ======
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

        # ====== C. 场景实体配置解析 ======
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

        # ====== D. 身体部位索引查找（G1 命名）======
        # G1 足部 link 名为 left_ankle_roll_link / right_ankle_roll_link
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        # G1 手腕末端 link：left/right_wrist_yaw_link
        # 直接用 wrist 真实位置作为 "hand"（不再用 elbow + 虚拟偏移）
        self.wrist_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            preserve_order=True,
        )

        # ====== E. 关节索引查找（G1 命名，对齐 AMP 数据的关节顺序）======
        # 关节顺序：yaw / roll / pitch / knee / ankle_pitch / ankle_roll
        # ↑ 与 g1_walkamp_*.txt 数据匹配（左 yaw 在最前）
        # 左腿 6 个关节
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
        # 右腿 6 个关节
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
        # 左臂 4 个关节（肘是 elbow_joint，不带 pitch 后缀）
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
            ],
            preserve_order=True,
        )
        # 右臂 4 个关节
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ],
            preserve_order=True,
        )
        # 4 个踝关节
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        # 腰部 3 关节（新增进 AMP obs，让 AMP 学到走路时的腰部微扭动）
        self.waist_joint_ids, _ = self.robot.find_joints(
            name_keys=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            preserve_order=True,
        )

        # ====== F. 观测缩放与噪声配置 ======
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        # ====== G. 运行时计数器 ======
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # ====== H. （已删）手臂末端虚拟偏移
        # 之前用 elbow + [0,0,-0.3] 近似 hand，现在直接用 wrist_yaw_link 真实位置 ======

        # ====== I. 步态时钟参数 ======
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

        # ====== 终态 AMP 观测缓存（用于修复 Reset 边界问题）======
        # AMP obs = 23 dof_pos + 23 dof_vel + 12 end_pos + 6 base_vel = 64
        self._amp_obs_dim = (
            2 * (len(self.left_arm_ids) + len(self.right_arm_ids)
                 + len(self.left_leg_ids) + len(self.right_leg_ids)
                 + len(self.waist_joint_ids))
            + 12  # 左手(3) + 右手(3) + 左脚(3) + 右脚(3)
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

        # ====== J. 动作与足部统计缓存 ======
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

        # ====== K. 初始化观测历史缓存 ======
        self.init_obs_buffer()

        # ====== L. RSI (Reference State Initialization) 加载 ======
        self._init_rsi()

    # ================================================================
    # AMP 观测计算（重写父类，按 G1 数据集的 left-first 顺序输出）
    # ================================================================
    def get_amp_obs_for_expert_trans(self):
        """
        计算 G1 当前时刻的 64 维 AMP 状态。

        数据布局（left-first 顺序）：
          dof_pos (23):
            [0:6]   left_leg, [6:12] right_leg, [12:15] waist, [15:19] left_arm, [19:23] right_arm
          dof_vel (23):       同上顺序
          end_pos (12):
            [46:49] left_hand (wrist_yaw_link), [49:52] right_hand
            [52:55] left_foot (ankle_roll_link), [55:58] right_foot
          base_vel (6):       root velocity in body frame
            [58:61] root_lin_vel_b, [61:64] root_ang_vel_b
        """
        body_state = self.robot.data.body_state_w
        root_pos = self.robot.data.root_state_w[:, 0:3]
        root_quat_inv = quat_conjugate(self.robot.data.root_state_w[:, 3:7])

        # ---- 末端位置：直接用真实 link 位置，转到 root 坐标系 ----
        def _pos_in_root(body_idx):
            rel_in_world = body_state[:, body_idx, :3] - root_pos
            return quat_apply(root_quat_inv, rel_in_world)

        left_hand_pos = _pos_in_root(self.wrist_body_ids[0])
        right_hand_pos = _pos_in_root(self.wrist_body_ids[1])
        left_foot_pos = _pos_in_root(self.feet_body_ids[0])
        right_foot_pos = _pos_in_root(self.feet_body_ids[1])

        # ---- 关节角度/速度（含 waist） ----
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

        # ---- 拼接 58 维 ----
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
                self.robot.data.root_lin_vel_b,  # [58:61] root 线速度（body frame）
                self.robot.data.root_ang_vel_b,  # [61:64] root 角速度（body frame）
            ),
            dim=-1,
        )

    # ================================================================
    # RSI (Reference State Initialization)
    # reset 时 rsi_prob 概率把 G1 spawn 到 expert 数据的随机一帧，
    # 让 policy 直接从"走路中段"开始练，加速训练 + 防判别器死锁。
    # ================================================================
    def _init_rsi(self):
        """加载 RSI reference state 数据并构造 mujoco→IsaacLab 关节映射。"""
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

        # 每帧字段切片（跟 extract_amp_data.py 一致）
        # [0:64]    amp_obs
        # [64:67]   root_pos (world)
        # [67:71]   root_quat (wxyz)
        # [71:100]  dof_pos_full (mujoco order, 29)
        # [100:129] dof_vel_full (mujoco order, 29)
        root_pos = frames[:, 64:67]
        root_quat = frames[:, 67:71]
        dof_pos_mj = frames[:, 71:100]
        dof_vel_mj = frames[:, 100:129]

        # 上传到 GPU
        self._rsi_root_pos = torch.tensor(root_pos, device=self.device)
        self._rsi_root_quat = torch.tensor(root_quat, device=self.device)
        # root vel 在 body frame，从 amp_obs 段提取（[58:61], [61:64]）
        self._rsi_root_lin_vel_b = torch.tensor(frames[:, 58:61], device=self.device)
        self._rsi_root_ang_vel_b = torch.tensor(frames[:, 61:64], device=self.device)

        # mujoco → IsaacLab 关节映射
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
        # 重排到 IsaacLab 顺序，提前存好（避免每次 reset 重排）
        self._rsi_dof_pos = torch.tensor(dof_pos_mj, device=self.device)[:, mj2lab]
        self._rsi_dof_vel = torch.tensor(dof_vel_mj, device=self.device)[:, mj2lab]

        self._rsi_n_frames = N
        self._rsi_prob = float(self.cfg.rsi_prob)
        self._rsi_enabled = True
        print(f"[INFO] ✅ RSI enabled: {N} reference frames, prob={self._rsi_prob}")

    def reset(self, env_ids):
        # 1. 标准 reset（default pose）
        super().reset(env_ids)

        if len(env_ids) == 0 or not getattr(self, "_rsi_enabled", False):
            return

        # 2. 随机决定哪些 env 用 RSI
        n = len(env_ids)
        use_rsi = torch.rand(n, device=self.device) < self._rsi_prob
        rsi_env_ids = env_ids[use_rsi]
        if len(rsi_env_ids) == 0:
            return

        # 3. 抽 expert 随机帧
        n_rsi = len(rsi_env_ids)
        frame_idx = torch.randint(0, self._rsi_n_frames, (n_rsi,), device=self.device)

        # 4. 写 root_state（13 维: pos 3 + quat 4 + lin_vel_w 3 + ang_vel_w 3）
        root_quat = self._rsi_root_quat[frame_idx]   # (n_rsi, 4) wxyz
        root_state = torch.zeros((n_rsi, 13), device=self.device, dtype=torch.float)
        # xy 用 env_origin（避免多 env 重叠），z 用 expert 高度
        root_state[:, 0:2] = self.scene.env_origins[rsi_env_ids, 0:2]
        root_state[:, 2] = self._rsi_root_pos[frame_idx, 2]
        root_state[:, 3:7] = root_quat
        # body frame vel → world frame vel
        lin_vel_b = self._rsi_root_lin_vel_b[frame_idx]
        ang_vel_b = self._rsi_root_ang_vel_b[frame_idx]
        root_state[:, 7:10] = quat_apply(root_quat, lin_vel_b)
        root_state[:, 10:13] = quat_apply(root_quat, ang_vel_b)
        self.robot.write_root_state_to_sim(root_state, rsi_env_ids)

        # 5. 写 dof_state（已经在 IsaacLab 顺序）
        dof_pos = self._rsi_dof_pos[frame_idx]
        dof_vel = self._rsi_dof_vel[frame_idx]
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=rsi_env_ids)
