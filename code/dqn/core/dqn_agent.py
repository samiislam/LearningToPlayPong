"""DQN agent: experience replay buffer and epsilon-greedy interaction loop."""
import collections
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from core import model as dqn_model


State = np.ndarray
Action = int
BatchTensors = tuple[
    torch.Tensor,               # current state
    torch.Tensor,               # actions
    torch.Tensor,               # rewards
    torch.Tensor,               # done || trunc
    torch.Tensor                # next state
]


@dataclass
class Experience:
    """A single one-step transition (s, a, r, done, s')."""
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    """Fixed-capacity replay buffer with uniform random sampling."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class Agent:
    """Interacts with vectorized environments using epsilon-greedy policy."""

    def __init__(self, env: gym.vector.VectorEnv, exp_buffer: ExperienceBuffer):
        self.env = env
        self.n_envs = env.num_envs
        self.exp_buffer = exp_buffer
        self.states: np.ndarray | None = None
        self.total_rewards = np.zeros(self.n_envs)
        self.total_steps = np.zeros(self.n_envs, dtype=int)
        self._reset()

    def _reset(self):
        self.states, _ = self.env.reset()
        self.total_rewards = np.zeros(self.n_envs)
        self.total_steps = np.zeros(self.n_envs, dtype=int)

    @torch.no_grad()
    def play_step(self, net: dqn_model.DQN, device: torch.device,
                  epsilon: float = 0.0) -> list[tuple[float, int]]:
        """Advance all envs by one step; return (reward, steps) for any finished episodes."""
        assert self.states is not None
        done_episodes: list[tuple[float, int]] = []

        # epsilon-greedy action selection
        if np.random.random() < epsilon:
            actions = self.env.action_space.sample()
        else:
            torch.compiler.cudagraph_mark_step_begin()
            states_v = torch.as_tensor(self.states).to(device)
            q_vals_v = net(states_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            actions = act_v.cpu().numpy()

        new_states, rewards, is_done, is_tr, infos = self.env.step(actions)
        self.total_rewards += rewards
        self.total_steps += 1

        for i in range(self.n_envs):
            done_trunc = bool(is_done[i]) or bool(is_tr[i])
            # on termination the auto-reset overwrites new_states[i],
            # so use final_observation to get the true last state
            if done_trunc and "final_observation" in infos:
                last_new_state = infos["final_observation"][i]
            else:
                last_new_state = new_states[i]
            exp = Experience(
                state=self.states[i], action=int(actions[i]), reward=float(rewards[i]),
                done_trunc=done_trunc, new_state=last_new_state
            )
            self.exp_buffer.append(exp)

            if done_trunc:
                done_episodes.append((float(self.total_rewards[i]), int(self.total_steps[i])))
                self.total_rewards[i] = 0.0
                self.total_steps[i] = 0

        self.states = new_states
        return done_episodes


def batch_to_tensors(batch: list[Experience], device: torch.device) -> BatchTensors:
    """Unpack a list of experiences into GPU-ready tensors."""
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return states_t.to(device, non_blocking=True), actions_t.to(device, non_blocking=True), \
           rewards_t.to(device, non_blocking=True), dones_t.to(device, non_blocking=True), \
           new_states_t.to(device, non_blocking=True)
