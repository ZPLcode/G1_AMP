# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under BSD-3-Clause.

"""Sim2Sim: deploy IsaacLab-trained G1 AMP policy to Mujoco.

Usage:
    python legged_lab/scripts/sim2sim_g1.py \\
        --policy logs/walk_amp_g1/<run_dir>/exported/policy.pt
"""

import argparse
import os
import sys
import time

import glfw
import mujoco
import mujoco_viewer
import numpy as np
import torch


MUJOCO_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

ISAAC_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

ISAAC_TO_MUJOCO_IDX = [ISAAC_JOINT_ORDER.index(n) for n in MUJOCO_JOINT_ORDER]
MUJOCO_TO_ISAAC_IDX = [MUJOCO_JOINT_ORDER.index(n) for n in ISAAC_JOINT_ORDER]


def _build_default_dof_pos():
    defaults = {
        "left_hip_pitch_joint": -0.20, "right_hip_pitch_joint": -0.20,
        "left_knee_joint": 0.42, "right_knee_joint": 0.42,
        "left_ankle_pitch_joint": -0.23, "right_ankle_pitch_joint": -0.23,
        "left_elbow_joint": 0.87, "right_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.18, "right_shoulder_roll_joint": -0.18,
        "left_shoulder_pitch_joint": 0.35, "right_shoulder_pitch_joint": 0.35,
    }
    return np.array([defaults.get(n, 0.0) for n in ISAAC_JOINT_ORDER], dtype=np.float32)


class SimToSimCfg:
    class sim:
        sim_duration = 60.0
        num_action = 29
        num_obs_per_step = 102
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        gait_air_ratio_l = 0.38
        gait_air_ratio_r = 0.38
        gait_phase_offset_l = 0.38
        gait_phase_offset_r = 0.88
        gait_cycle = 0.85


class MujocoG1Runner:
    def __init__(self, cfg, policy_path, model_path):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = cfg.sim.dt
        self.data = mujoco.MjData(self.model)

        n_act = self.model.nu
        for i in range(n_act):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            assert jnt_name == MUJOCO_JOINT_ORDER[i], (
                f"Actuator {i}={jnt_name}, expected {MUJOCO_JOINT_ORDER[i]}"
            )
        print(f"[INFO] {n_act} position actuators loaded, joint order verified")

        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"[INFO] policy loaded: {policy_path}")

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False

        self.dt = cfg.sim.decimation * cfg.sim.dt
        self.dof_pos = np.zeros(cfg.sim.num_action, dtype=np.float32)
        self.dof_vel = np.zeros(cfg.sim.num_action, dtype=np.float32)
        self.action = np.zeros(cfg.sim.num_action, dtype=np.float32)
        self.default_dof_pos = _build_default_dof_pos()
        self.episode_length_buf = 0

        self.gait_phase = np.zeros(2, dtype=np.float32)
        self.phase_ratio = np.array(
            [cfg.robot.gait_air_ratio_l, cfg.robot.gait_air_ratio_r], dtype=np.float32
        )
        self.phase_offset = np.array(
            [cfg.robot.gait_phase_offset_l, cfg.robot.gait_phase_offset_r], dtype=np.float32
        )

        self.command_vel = np.zeros(3, dtype=np.float32)

        self.obs_history = np.zeros(
            cfg.sim.num_obs_per_step * cfg.sim.actor_obs_history_length, dtype=np.float32
        )

        default_mujoco = self.default_dof_pos[ISAAC_TO_MUJOCO_IDX]
        self.data.qpos[0:3] = [0.0, 0.0, 0.80]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:36] = default_mujoco
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        qpos_mujoco = self.data.qpos[7:36].copy()
        qvel_mujoco = self.data.qvel[6:35].copy()
        self.dof_pos = qpos_mujoco[MUJOCO_TO_ISAAC_IDX].astype(np.float32)
        self.dof_vel = qvel_mujoco[MUJOCO_TO_ISAAC_IDX].astype(np.float32)

        ang_vel = self.data.qvel[3:6].astype(np.float32)
        quat_wxyz = self.data.qpos[3:7].astype(np.float32)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        proj_grav = self.quat_rotate_inverse(quat_xyzw, np.array([0.0, 0.0, -1.0]))

        obs = np.concatenate([
            ang_vel,
            proj_grav,
            self.command_vel,
            self.dof_pos - self.default_dof_pos,
            self.dof_vel,
            np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),
            np.sin(2 * np.pi * self.gait_phase),
            np.cos(2 * np.pi * self.gait_phase),
            self.phase_ratio,
        ]).astype(np.float32)

        self.obs_history = np.roll(self.obs_history, -self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step:] = obs

        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def compute_target_dof_pos_mujoco(self):
        target_pos = self.action * self.cfg.sim.action_scale + self.default_dof_pos
        return target_pos[ISAAC_TO_MUJOCO_IDX].astype(np.float32)

    def run(self):
        self.setup_keyboard_listener()

        target_dof_pos = self.default_dof_pos[ISAAC_TO_MUJOCO_IDX].astype(np.float32)
        counter = 0

        while self.data.time < self.cfg.sim.sim_duration:
            step_start = time.time()

            self.data.ctrl[:] = target_dof_pos
            mujoco.mj_step(self.model, self.data)
            self.viewer.render()

            counter += 1
            if counter % self.cfg.sim.decimation == 0:
                obs = self.get_obs()
                with torch.inference_mode():
                    action_tensor = self.policy(torch.tensor(obs, dtype=torch.float32))
                self.action = action_tensor.detach().numpy()[:self.cfg.sim.num_action].astype(np.float32)
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                target_dof_pos = self.compute_target_dof_pos_mujoco()

                self.episode_length_buf += 1
                self.calculate_gait_para()

            elapsed = time.time() - step_start
            if (sleep := self.cfg.sim.dt - elapsed) > 0:
                time.sleep(sleep)

        self.viewer.close()

    @staticmethod
    def quat_rotate_inverse(q_xyzw, v):
        q_w = q_xyzw[-1]
        q_vec = q_xyzw[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def calculate_gait_para(self):
        t = self.episode_length_buf * self.dt / self.cfg.robot.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def setup_keyboard_listener(self):
        glfw.set_key_callback(self.viewer.window, self.key_callback)

    def key_callback(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        STEP = 0.05
        if key in (glfw.KEY_W, glfw.KEY_UP):    self.command_vel[0] += STEP
        elif key in (glfw.KEY_S, glfw.KEY_DOWN): self.command_vel[0] -= STEP
        elif key in (glfw.KEY_A, glfw.KEY_LEFT): self.command_vel[1] += STEP
        elif key in (glfw.KEY_D, glfw.KEY_RIGHT):self.command_vel[1] -= STEP
        elif key == glfw.KEY_Q:                  self.command_vel[2] += STEP
        elif key == glfw.KEY_E:                  self.command_vel[2] -= STEP
        elif key == glfw.KEY_SPACE:              self.command_vel[:] = 0
        self.command_vel = np.clip(self.command_vel, -1.0, 1.0)
        print(f"[CMD] vx={self.command_vel[0]:+.2f} vy={self.command_vel[1]:+.2f} wz={self.command_vel[2]:+.2f}")


if __name__ == "__main__":
    LEGGED_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to exported policy.pt (TorchScript)")
    parser.add_argument("--model", type=str,
                        default=os.path.join(
                            LEGGED_LAB_ROOT,
                            "legged_lab/assets/unitree/g1/mjcf/g1_29dof_rev_1_0_daf.xml"))
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    if not os.path.isfile(args.policy):
        sys.exit(f"[ERROR] policy not found: {args.policy}")
    if not os.path.isfile(args.model):
        sys.exit(f"[ERROR] model not found: {args.model}")

    print(f"[INFO] policy: {args.policy}")
    print(f"[INFO] model:  {args.model}")

    cfg = SimToSimCfg()
    cfg.sim.sim_duration = args.duration

    runner = MujocoG1Runner(cfg, args.policy, args.model)
    runner.run()
