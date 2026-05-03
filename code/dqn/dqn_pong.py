#!/usr/bin/env python3
"""Deep Q-Network (DQN) on Pong.

Off-policy algorithm that learns a Q-value function from a replay buffer
of past transitions.  Uses a target network (Polyak-averaged) for stable
bootstrap targets and epsilon-greedy exploration.
"""
import argparse
import time
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
from core.dqn_agent import Agent, Experience, ExperienceBuffer, batch_to_tensors


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
