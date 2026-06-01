# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
复杂地形 + 高度扫描行走实验配置

设计动机：
    项目原本所有任务都在 GRAVEL_TERRAINS_CFG（仅 ±2cm 噪声）上训练。
    代码里其实定义了 ROUGH_TERRAINS_CFG（楼梯/方块/波浪/坑洞），但从未启用。
    本实验：
      1. 启用 ROUGH 复杂地形
      2. 启用 height_scanner（机器人能"看到"前方地形）
      3. 继承 walk_with_sensor 的 AMP 配置

预期：
    - 训练时间比 walk 长 2-3 倍（地形难度高）
    - episode_length 可能比平地低（更容易摔）
    - height_scan 让 actor obs 维度从 75 升到 235，网络要重新训练
    - 视觉效果好，可生成上楼梯的演示视频
"""

from isaaclab.utils import configclass

from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
from legged_lab.terrains import ROUGH_TERRAINS_CFG


@configclass
class TienKungWalkTerrainFlatEnvCfg(TienKungWalkWithSensorFlatEnvCfg):
    """复杂地形 + 高度扫描配置"""

    def __post_init__(self):
        # 1. 切换到 ROUGH 复杂地形（含楼梯/方块/波浪/坑洞）
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG

        # 2. 启用高度扫描器（机器人能"看到"前方 1.6m × 1.0m 地形）
        #    扫描分辨率 0.1m → 16 × 10 = 160 个高度值
        #    会让 actor 观测维度从 75 → 235
        self.scene.height_scanner.enable_height_scan = True

        # 3. 关闭 LiDAR 和深度相机（很关键！）
        #    walk_with_sensor 同时启用了 LiDAR + DepthCamera，二者都用 TiledCamera
        #    每个 TiledCamera 在 4096 个环境下要占约 2GB 显存 → 16GB GPU 会爆
        #    本实验只用 height_scanner，把这两个相机关掉
        self.scene.lidar.enable_lidar = False
        self.scene.depth_camera.enable_depth_camera = False

        # 4. 调整奖励权重适应复杂地形（核心改动）
        # ──────────────────────────────────────────────────
        # 平地训练用的姿态约束在楼梯/斜面上会"冲突"：
        #   原始权重逼着躯干必须直立 → 机器人不敢上楼梯（怕扣分）
        # 调整原则：放宽姿态和冲击力相关惩罚，让策略敢挑战地形

        # 4.1 躯干姿态：放宽 4 倍（允许上下台阶时身体倾斜）
        self.reward.body_orientation_l2.weight = -0.5  # 原 -2.0
        self.reward.flat_orientation_l2.weight = -0.3  # 原 -1.0

        # 4.2 踩地冲击力：放宽 3 倍（楼梯下台阶冲击天然更大）
        self.reward.feet_force.weight = -1e-3  # 原 -3e-3

        # 4.3 绊倒惩罚：放宽 4 倍（不平地面易误判为绊倒）
        self.reward.feet_stumble.weight = -0.5  # 原 -2.0


@configclass
class TienKungWalkTerrainAgentCfg(TienKungWalkWithSensorAgentCfg):
    """复杂地形 Agent 配置：继承 walk_with_sensor 的 AMP 配置，只改实验名"""

    def __post_init__(self):
        self.experiment_name = "walk_terrain_sensor"
        self.neptune_project = "walk_terrain_sensor"
        self.wandb_project = "walk_terrain_sensor"
