# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
基于原 g1_walkamp_B1_50fps.txt 重新生成 AMP 训练数据。

核心思路：
    end_pos 12 维（hand/foot 在 root 坐标系下的位置）原本是用 G1DWAQ_Lab fork
    的某个公式算的，跟我们 env 的 get_amp_obs_for_expert_trans() 可能不一致。
    本脚本做法：
        - dof_pos: 用原数据（保持真实走路姿态）
        - end_pos: 用我们 env 的公式实时算（保证训练时 expert 和 policy obs 同公式）
        - dof_vel: 用相邻帧 dof_pos 的有限差分（一致的来源）

用法:
    python legged_lab/scripts/regenerate_amp_data.py
"""

import argparse
import json

import numpy as np
from isaaclab.app import AppLauncher

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Regenerate G1 AMP data with our env's amp_obs formula.")
parser.add_argument("--task", type=str, default="walk_amp_g1")
parser.add_argument(
    "--motion_file",
    type=str,
    default="legged_lab/envs/g1/datasets/g1_walkamp_B1_50fps.txt",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="legged_lab/envs/g1/datasets/g1_walkamp_B1_50fps_regen.txt",
)
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 强制 headless（不需要 GUI）
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def main():
    # ====== 1. 加载原数据 ======
    with open(args_cli.motion_file) as f:
        data = json.load(f)
    frames = np.array(data["Frames"])  # (N, 52)
    dt = float(data["FrameDuration"])
    N = len(frames)
    print(f"\n[INFO] Loaded {N} frames @ {1/dt:.1f}FPS from {args_cli.motion_file}")

    # ====== 2. 启动 env（单 robot, plane terrain, 无噪声）======
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.commands.debug_vis = False

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    device = env.device

    print(f"[INFO] Env loaded. Joints: {env.robot.num_joints}")
    print(f"[INFO] left_leg_ids:  {env.left_leg_ids}")
    print(f"[INFO] right_leg_ids: {env.right_leg_ids}")
    print(f"[INFO] left_arm_ids:  {env.left_arm_ids}")
    print(f"[INFO] right_arm_ids: {env.right_arm_ids}")

    # ====== 3. 固定 root state（避免重力拉倒）======
    fixed_root_state = torch.zeros((1, 13), device=device)
    fixed_root_state[:, 2] = 0.8  # z 高度
    fixed_root_state[:, 3] = 1.0  # quat_w（无旋转）
    env_ids_all = torch.arange(1, device=device)

    # ====== 4. 逐帧重新算 amp_obs ======
    dof_pos_template = env.robot.data.default_joint_pos.clone()
    dof_vel_template = torch.zeros_like(dof_pos_template)
    new_frames = np.zeros((N, 52), dtype=np.float32)

    print(f"\n[INFO] Regenerating {N} frames...")
    for i in range(N):
        if not simulation_app.is_running():
            print("[WARN] Sim app closed early.")
            break

        frame = torch.tensor(frames[i], device=device, dtype=torch.float)

        # 把原数据的 dof_pos 写到 robot
        dof_pos = dof_pos_template.clone()
        dof_pos[:, env.left_leg_ids] = frame[0:6]
        dof_pos[:, env.right_leg_ids] = frame[6:12]
        dof_pos[:, env.left_arm_ids] = frame[12:16]
        dof_pos[:, env.right_arm_ids] = frame[16:20]

        env.robot.write_joint_state_to_sim(dof_pos, dof_vel_template)
        env.robot.write_root_state_to_sim(fixed_root_state, env_ids_all)

        # Step 让 body_state_w 更新（hand_pos 计算需要 elbow 的世界位姿）
        env.sim.step()
        env.scene.update(dt=dt)

        # 调 env 的公式算 amp_obs（保证跟训练时同一公式）
        amp_obs = env.get_amp_obs_for_expert_trans()
        new_frames[i] = amp_obs[0].cpu().numpy()

        if i % 50 == 0:
            print(f"  Frame {i}/{N}")

    # ====== 5. 用相邻帧 dof_pos 的有限差分覆盖 dof_vel ======
    # 这样 vel 和 pos 严格一致（不像原数据 vel 不知道哪来的）
    pos = new_frames[:, 0:20]  # (N, 20)
    vel = np.zeros_like(pos)
    vel[:-1] = (pos[1:] - pos[:-1]) / dt
    vel[-1] = vel[-2]  # 最后一帧复制
    new_frames[:, 20:40] = vel

    # ====== 6. 写新 JSON ======
    out_data = {
        "LoopMode": data.get("LoopMode", "Wrap"),
        "FrameDuration": dt,
        "EnableCycleOffsetPosition": data.get("EnableCycleOffsetPosition", True),
        "EnableCycleOffsetRotation": data.get("EnableCycleOffsetRotation", True),
        "MotionWeight": data.get("MotionWeight", 0.5),
        "Frames": new_frames.tolist(),
    }
    with open(args_cli.output_file, "w") as f:
        json.dump(out_data, f)

    print(f"\n[INFO] ✅ Saved to {args_cli.output_file}")
    print(f"  dof_pos range: [{new_frames[:, :20].min():.3f}, {new_frames[:, :20].max():.3f}]")
    print(f"  dof_vel range: [{new_frames[:, 20:40].min():.3f}, {new_frames[:, 20:40].max():.3f}]")
    print(f"  end_pos range: [{new_frames[:, 40:52].min():.3f}, {new_frames[:, 40:52].max():.3f}]")
    print(f"\n  First frame end_pos breakdown:")
    print(f"    left_hand:  {new_frames[0, 40:43]}")
    print(f"    right_hand: {new_frames[0, 43:46]}")
    print(f"    left_foot:  {new_frames[0, 46:49]}")
    print(f"    right_foot: {new_frames[0, 49:52]}")
    print(f"\n  原数据 frame 0 end_pos （对比）:")
    print(f"    left_hand:  {frames[0, 40:43]}")
    print(f"    right_hand: {frames[0, 43:46]}")
    print(f"    left_foot:  {frames[0, 46:49]}")
    print(f"    right_foot: {frames[0, 49:52]}")


if __name__ == "__main__":
    main()
    simulation_app.close()
