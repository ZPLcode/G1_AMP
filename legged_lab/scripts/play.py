# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# ====== 1. 解析命令行参数与环境设置 ======
# 添加标准命令行参数
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# 追加 RSL-RL 与 AppLauncher 专属命令行参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 如果是传感器相关任务，强制开启相机渲染流
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# ====== 2. 启动 Omniverse Isaac Sim 仿真底座 ======
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    """
    策略仿真评估与导出主函数。
    加载训练好的权重，净化测试环境，导出通用部署模型，并运行纯推断测试。
    """
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    # ====== 3. 加载任务配置并净化环境参数 ======
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # 关闭影响观感的扰动与随机化，以便观察最纯粹的步态姿态
    env_cfg.noise.add_noise = True                          # [DEBUG] 跟训练一致保留 noise
    # env_cfg.domain_rand.events.push_robot = None         # [DEBUG] 跟训练一致：保留 push_robot
    env_cfg.scene.max_episode_length_s = 10.0               # 将测试单局最大时长延长至 40 秒
    env_cfg.scene.num_envs = 50                             # 减少并行环境为 50，以保证 GUI 渲染帧率极度丝滑
    env_cfg.scene.env_spacing = 2.5                         # 设置环境间的初始间距
    env_cfg.commands.rel_standing_envs = 0.0                # 测试时所有机器人保持行走，不设置静止个体
    env_cfg.commands.ranges.lin_vel_x = (0.5, 0.5)          # 0.5 m/s（接近训练 command 分布 mean）
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)          # 强制目标侧向线速度为 0.0 m/s
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)   # 消除高度扫描仪的漂移误差

    # [DEBUG] 跟训练一致：保留 GRAVEL_TERRAINS_CFG，不改 plane
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    # 若命令行覆盖了并行环境数，进行覆盖
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # 更新训练配置与随机种子
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    # play 时关 RSI，让 G1 从 default pose 起步（避免 spawn 在 expert 中段）
    env_cfg.rsi_enable = False
    env_cfg.rsi_prob = 0.0

    # ====== 4. 实例化环境 ======
    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    # [DEBUG] 打印 robot 实际 joint_names 顺序（用于 sim2sim joint mapping 验证）
    joint_names = env.robot.data.joint_names
    print(f"\n[DEBUG] G1 joint count: {len(joint_names)}")
    print("[DEBUG] joint_names (Isaac order, used by policy obs):")
    for i, n in enumerate(joint_names):
        print(f"  {i:2d}: {n}")
    print()

    # ====== 5. 加载已保存的 Checkpoint 权重 ======
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # 实例化算法 Runner 并加载模型参数（加载权重但不加载优化器参数，因为这里不进行训练）
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    # 获取用于推断的纯前向策略网络接口
    policy = runner.get_inference_policy(device=env.device)

    # ====== 6. 导出可用于实机/Sim2Sim部署的通用格式模型 ======
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # 导出为 TorchScript JIT 二进制模型 (已自动打包观测归一化参数 obs_normalizer)
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # 导出为 ONNX 通用模型文件
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # ====== 7. 注册键盘实时交互控制器 ======
    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841

    # 获取第一帧初始状态观测
    obs, _ = env.get_observations()

    # ====== 8. 开始纯前向推理大循环 ======
    _step_counter = 0
    while simulation_app.is_running():
        # inference_mode 块不计算梯度，极大提升推理计算效率
        with torch.inference_mode():
            actions = policy(obs)             # 前向输入观测，得到动作指令
            # [DEBUG-SIM2SIM] 第一帧 dump 最新一帧 obs（最后 102 维）+ action
            # 用于跟 sim2sim_g1.py get_obs 的输出对比
            if _step_counter == 0:
                latest = obs[0, -102:].cpu().numpy()
                print("\n[DEBUG-SIM2SIM] first frame obs breakdown (102 dim):")
                print(f"  ang_vel    [0:3]   = {latest[0:3].round(3)}")
                print(f"  proj_grav  [3:6]   = {latest[3:6].round(3)}")
                print(f"  cmd        [6:9]   = {latest[6:9].round(3)}")
                print(f"  dof_pos_rel[9:38]  = {latest[9:38].round(3)}")
                print(f"  dof_vel    [38:67] = {latest[38:67].round(3)}")
                print(f"  last_act   [67:96] = {latest[67:96].round(3)}")
                print(f"  sin_phase  [96:98] = {latest[96:98].round(3)}")
                print(f"  cos_phase  [98:100]= {latest[98:100].round(3)}")
                print(f"  phase_rat [100:102]= {latest[100:102].round(3)}")
                print(f"  action.first = {actions[0].cpu().numpy().round(3)}")
                print()
            _step_counter += 1
            obs, _, _, _ = env.step(actions)  # 物理环境应用动作指令并前进一步，反馈新观测


if __name__ == "__main__":
    play()
    simulation_app.close()
