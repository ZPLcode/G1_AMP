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

import numpy as np
import torch


class ReplayBuffer:
    """AMP 专用经验回放缓存，存储策略历史状态转移 (s_t, s_{t+1})，用于训练判别器。

    与 RolloutStorage 的核心区别：
    - RolloutStorage 每轮清空；ReplayBuffer 跨轮保留，循环覆盖旧数据
    - ReplayBuffer 随机有放回采样，不依赖时序连续性
    - 混入历史各阶段策略数据，防止判别器因训练分布突变而震荡
    """

    def __init__(self, obs_dim, buffer_size, device, cmd_dim=3):
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.cmds = torch.zeros(buffer_size, cmd_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(self, states, next_states, cmds=None):
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if cmds is None:
            cmds = torch.zeros(num_states, self.cmds.shape[1], device=states.device)
        if end_idx > self.buffer_size:
            wrap = self.buffer_size - self.step
            self.states[self.step : self.buffer_size] = states[:wrap]
            self.next_states[self.step : self.buffer_size] = next_states[:wrap]
            self.cmds[self.step : self.buffer_size] = cmds[:wrap]
            self.states[: end_idx - self.buffer_size] = states[wrap:]
            self.next_states[: end_idx - self.buffer_size] = next_states[wrap:]
            self.cmds[: end_idx - self.buffer_size] = cmds[wrap:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states
            self.cmds[start_idx:end_idx] = cmds
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """vanilla：返回 (s, s_next)。"""
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device), self.next_states[sample_idxs].to(self.device))

    def feed_forward_generator_with_cmds(self, num_mini_batch, mini_batch_size):
        """conditional：返回 (s, s_next, cmds)。"""
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (
                self.states[sample_idxs].to(self.device),
                self.next_states[sample_idxs].to(self.device),
                self.cmds[sample_idxs].to(self.device),
            )
