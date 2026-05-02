#!/usr/bin/env python3
"""Deep Q-Network (DQN) on Pong.

Off-policy algorithm that learns a Q-value function from a replay buffer
of past transitions.  Uses a target network (Polyak-averaged) for stable
bootstrap targets and epsilon-greedy exploration.
"""
import argparse
import collections
import time
from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

from core import model as dqn_model
from core import wrappers


BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-4
MEAN_REWARD_BOUND = 19
N_ENVS = 8
REPLAY_SIZE = 100000
REPLAY_START_SIZE = 10000   # fill buffer before training starts
SEED = 42
TAU = 0.005                 # Polyak averaging coefficient for target net

EPSILON_DECAY_LAST_FRAME = 150000 * N_ENVS * 0.75
EPSILON_FINAL = 0.01
EPSILON_START = 1.0

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

    def __init__(self, env: gym.vector.VectorEnv, exp_buffer: ExperienceBuffer,
                 gamma: float = GAMMA):
        self.env = env
        self.exp_buffer = exp_buffer
        self.gamma = gamma
        self.states: np.ndarray | None = None
        self.total_rewards = np.zeros(N_ENVS)
        self.total_steps = np.zeros(N_ENVS, dtype=int)
        self._reset()

    def _reset(self):
        self.states, _ = self.env.reset()
        self.total_rewards = np.zeros(N_ENVS)
        self.total_steps = np.zeros(N_ENVS, dtype=int)

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

        for i in range(N_ENVS):
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


def calc_loss(batch: list[Experience], net: dqn_model.DQN, tgt_net: dqn_model.DQN,
              device: torch.device, gamma: float = GAMMA) -> torch.Tensor:
    """Compute the DQN temporal-difference loss.

    Uses the target network for bootstrap value estimation:
    Q_target = r + gamma * max_a' Q_tgt(s', a')   (0 if terminal).
    """
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    with torch.autocast(device_type=device.type):
        state_action_values = net(states_t).gather(
            1, actions_t.unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            next_state_values = tgt_net(new_states_t).max(1)[0]
            next_state_values[dones_t] = 0.0

        expected_state_action_values = next_state_values * gamma + rewards_t
        return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda", help="Device name, default=cuda")
    args = parser.parse_args()
    device = torch.device(args.dev)

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    env_factories = [
        lambda: wrappers.make_env(gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0))
        for _ in range(N_ENVS)
    ]
    env = gym.vector.AsyncVectorEnv(env_factories)
    assert isinstance(env.single_observation_space, gym.spaces.Box)
    assert isinstance(env.single_action_space, gym.spaces.Discrete)
    raw_net = dqn_model.DQN(env.single_observation_space.shape, env.single_action_space.n).to(device)
    net = cast(dqn_model.DQN, torch.compile(raw_net, backend="cudagraphs"))
    tgt_net = cast(dqn_model.DQN, torch.compile(
        dqn_model.DQN(env.single_observation_space.shape, env.single_action_space.n).to(device),
        backend="cudagraphs"))
    writer = SummaryWriter(comment="-pong-dqn")
    print(net)
    print(f"Actions: {env.single_action_space.n}")

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler("cuda")
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_ts = ts
    best_m_reward = None
    solved = False
    speed = 0.0

    while not solved:
        frame_idx += N_ENVS
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        episodes = agent.play_step(net, device, epsilon)
        # update speed estimate when episodes finish
        if episodes:
            now = time.time()
            elapsed = now - ts
            if elapsed > 0:
                speed = (frame_idx - ts_frame) / elapsed
            ts_frame = frame_idx
            ts = now
        for reward, steps in episodes:
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_ts))
            print(f"{elapsed} {frame_idx}: done {len(total_rewards)} games, "
                  f"reward {m_reward:.3f}, eps {epsilon:.2f}, speed {speed:.2f} f/s")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("steps", steps, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(raw_net.state_dict(), "dqn-model-best.dat")
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                solved = True
                break

        # wait until buffer has enough samples before training
        if len(buffer) < REPLAY_START_SIZE:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        scaler.scale(loss_t).backward()
        scaler.step(optimizer)
        scaler.update()

        # Polyak averaging: soft-update target network
        with torch.no_grad():
            for p, p_tgt in zip(net.parameters(), tgt_net.parameters()):
                p_tgt.data.mul_(1 - TAU).add_(TAU * p.data)
    writer.close()
