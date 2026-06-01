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

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.g1.g1_env import TienKungG1Env
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_ppo_cfg import (
    TienKungWalkPPOAgentCfg,
    TienKungWalkPPOFlatEnvCfg,
)
from legged_lab.envs.tienkung.run_ppo_fast_cfg import (
    TienKungRunPpoFastAgentCfg,
    TienKungRunPpoFastFlatEnvCfg,
)
from legged_lab.envs.g1.walk_ppo_g1_cfg import (
    TienKungWalkPPOG1AgentCfg,
    TienKungWalkPPOG1FlatEnvCfg,
)
from legged_lab.envs.g1.walk_amp_g1_cfg import (
    TienKungWalkAMPG1AgentCfg,
    TienKungWalkAMPG1FlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_terrain_cfg import (
    TienKungWalkTerrainAgentCfg,
    TienKungWalkTerrainFlatEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register(
    "walk_ppo", TienKungEnv, TienKungWalkPPOFlatEnvCfg(), TienKungWalkPPOAgentCfg()
)
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "run_ppo_fast", TienKungEnv, TienKungRunPpoFastFlatEnvCfg(), TienKungRunPpoFastAgentCfg()
)
task_registry.register(
    "walk_ppo_g1", TienKungG1Env, TienKungWalkPPOG1FlatEnvCfg(), TienKungWalkPPOG1AgentCfg()
)
task_registry.register(
    "walk_amp_g1", TienKungG1Env, TienKungWalkAMPG1FlatEnvCfg(), TienKungWalkAMPG1AgentCfg()
)
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "walk_terrain_sensor", TienKungEnv, TienKungWalkTerrainFlatEnvCfg(), TienKungWalkTerrainAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)

