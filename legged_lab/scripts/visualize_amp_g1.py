# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.


import argparse
import json

import numpy as np
from isaaclab.app import AppLauncher

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Visualize G1 AMP motion data.")
parser.add_argument("--task", type=str, default="walk_amp_g1", help="Task name (for env setup)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default 1)")
parser.add_argument("--seed", type=int, default=None, help="Seed (passed to env)")
parser.add_argument(
    "--motion_file",
    type=str,
    default="legged_lab/envs/g1/datasets/g1_walkamp_B1_50fps.txt",
    help="Path to G1 AMP motion data (.txt JSON)",
)
parser.add_argument("--loop", type=int, default=3, help="Number of times to loop the animation")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def main():
    with open(args_cli.motion_file) as f:
        data = json.load(f)
    frames = np.array(data["Frames"])  # (N, 52)
    dt = data["FrameDuration"]
    print(f"\n[INFO] Loaded {len(frames)} frames, FPS={1/dt:.1f}, "
          f"shape={frames.shape}")
    print(f"[INFO] 假设格式：[0:20]=dof_pos, [20:40]=dof_vel, [40:52]=end_pos\n")

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

    print(f"\n[INFO] 机器人关节数: {env.robot.num_joints}")
    print(f"[INFO] left_leg_ids:   {env.left_leg_ids}")
    print(f"[INFO] right_leg_ids:  {env.right_leg_ids}")
    print(f"[INFO] waist_joint_ids:{env.waist_joint_ids}")
    print(f"[INFO] left_arm_ids:   {env.left_arm_ids}")
    print(f"[INFO] right_arm_ids:  {env.right_arm_ids}\n")

    device = env.device
    dof_pos_template = env.robot.data.default_joint_pos.clone()
    dof_vel_template = torch.zeros_like(dof_pos_template)

    fixed_root_state = torch.zeros((1, 13), device=device)
    fixed_root_state[:, 2] = 0.8
    fixed_root_state[:, 3] = 1.0
    env_ids_all = torch.arange(1, device=device)

    print("[INFO] Starting playback...")
    for loop_idx in range(args_cli.loop):
        print(f"\n--- Loop {loop_idx + 1}/{args_cli.loop} ---")
        for frame_idx in range(len(frames)):
            if not simulation_app.is_running():
                return

            frame = torch.tensor(frames[frame_idx], device=device, dtype=torch.float)

            # [0:6] left_leg | [6:12] right_leg | [12:15] waist | [15:19] left_arm | [19:23] right_arm
            dof_pos = dof_pos_template.clone()
            dof_vel = dof_vel_template.clone()

            dof_pos[:, env.left_leg_ids] = frame[0:6]
            dof_pos[:, env.right_leg_ids] = frame[6:12]
            dof_pos[:, env.waist_joint_ids] = frame[12:15]
            dof_pos[:, env.left_arm_ids] = frame[15:19]
            dof_pos[:, env.right_arm_ids] = frame[19:23]

            _ = frame[23:46]  # silence unused

            env.robot.write_joint_state_to_sim(dof_pos, dof_vel)
            env.robot.write_root_state_to_sim(fixed_root_state, env_ids_all)

            env.sim.render()
            env.sim.step()
            env.scene.update(dt=dt)

            if frame_idx % int(1.0 / dt) == 0:
                print(f"  Frame {frame_idx}/{len(frames)}")

    print("\n[INFO] Playback finished. Close window to exit.")
    while simulation_app.is_running():
        env.sim.render()


if __name__ == "__main__":
    main()
    simulation_app.close()
