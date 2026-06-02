# TienKung-Lab (Unitree G1 Extension)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL RL](https://img.shields.io/badge/RSL_RL-2.3.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

Personal fork of [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) extending the
AMP-PPO locomotion framework to the **Unitree G1 (29-DoF)** humanoid, with end-to-end
Mujoco sim2sim deployment.

## What's added on top of the original

| Module | Original (TienKung 20-DoF) | This fork (G1 29-DoF) |
|---|---|---|
| Robot | TienKung | G1 (waist 3 + wrist 6) |
| Env config | `envs/tienkung/walk_cfg.py` | `envs/g1/walk_ppo_g1_cfg.py`, `envs/g1/walk_amp_g1_cfg.py` |
| Env class | `TienKungEnv` | `TienKungG1Env` (overrides joint/body naming) |
| Expert data | LAFAN1 → TienKung | LAFAN1 → G1 (via [GMR](https://github.com/YanjieZe/GMR)) |
| Sim2sim | `scripts/sim2sim.py` | `scripts/sim2sim_g1.py` (G1 obs layout + joint mapping) |
| Mujoco asset | `assets/tienkung2_lite/` | `assets/unitree/g1/mjcf/` (with `<position>` actuators) |

Conditional AMP sampling (`rsl_rl/utils/motion_loader.py`) selects expert frames by command
direction, and Reference State Initialization (`envs/g1/g1_env.py`) lets training spawn
from arbitrary expert frames instead of default pose.

## Installation

Same as the upstream project. Built with IsaacSim 4.5.0 + IsaacLab 2.1.0.

```bash
# Inside a conda env with IsaacLab installed
cd TienKung-Lab && pip install -e .
cd TienKung-Lab/rsl_rl && pip install -e .
```

Verify the install:

```bash
python legged_lab/scripts/train.py --task=walk --headless --num_envs=64
```

## Usage

### Train G1

```bash
# Pure PPO baseline
python legged_lab/scripts/train.py --task=walk_ppo_g1 --headless --num_envs=4096

# AMP-PPO (with LAFAN1 expert mocap)
python legged_lab/scripts/train.py --task=walk_amp_g1 --headless --num_envs=4096
```

### Play (IsaacLab visualization + export)

```bash
python legged_lab/scripts/play.py --task=walk_amp_g1 --num_envs=1
```

Running `play.py` exports the latest checkpoint to
`logs/walk_amp_g1/<run>/exported/policy.pt` (TorchScript) and `policy.onnx`.

### Sim2Sim (Mujoco)

```bash
python legged_lab/scripts/sim2sim_g1.py \
    --policy logs/walk_amp_g1/<run>/exported/policy.pt
```

In the Mujoco viewer, use keyboard to drive the command:
`W/S` adjusts `vx`, `A/D` adjusts `vy`, `Q/E` adjusts `wz`, `SPACE` resets to zero.

The Mujoco model (`assets/unitree/g1/mjcf/g1_29dof_rev_1_0_daf.xml`) uses `<position>`
actuators with `kp`/`kv` matched to the IsaacLab `G1_CFG.actuators` stiffness/damping.

### Original TienKung tasks

The original TienKung walk/run tasks still work; see the
[upstream README](https://github.com/Open-X-Humanoid/TienKung-Lab) for retargeting and
sensor-based training details.

## Implementation notes

- **Joint order mismatch**: IsaacLab orders G1 joints by USD-tree depth (`left_hip_pitch,
  right_hip_pitch, waist_yaw, left_hip_roll, ...`), Mujoco orders them by `<actuator>` tag
  position. `sim2sim_g1.py` keeps both `ISAAC_JOINT_ORDER` and `MUJOCO_JOINT_ORDER`
  constants and re-shuffles obs/action across the boundary.

## Acknowledgements

- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) — upstream codebase.
- [GMR](https://github.com/YanjieZe/GMR) — general motion retargeting (LAFAN1 → G1).
- [Legged Lab](https://github.com/Hellod035/LeggedLab) — direct IsaacLab workflow.
- [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) — RL framework with sim2sim.
- [RSL RL](https://github.com/leggedrobotics/rsl_rl) — RL algorithms.
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware) — AMP reference implementation.
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) — G1 PD reference / deploy framework.
