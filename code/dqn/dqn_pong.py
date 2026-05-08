"""Deep Q-Network (DQN) on Pong.

Implements "Algorithm 1: deep Q-learning with experience replay" from
Mnih et al. 2015, "Human-level control through deep reinforcement learning"
(Nature 518, 529-533). Off-policy: learns the greedy policy
a = argmax_a' Q(s, a'; theta) while behaving epsilon-greedy.

Deviations from the paper:
  - Soft Polyak target update (TAU per step) instead of a hard copy every C steps.
  - MSE loss instead of the [-1, 1]-clipped error (Huber-equivalent) the paper uses.
  - Vectorised envs (N_ENVS) collect transitions in parallel.
  - Adam optimiser instead of RMSProp.
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
    """Compute the DQN TD loss on a sampled minibatch (Algorithm 1, target y_j).

    Implements:
        y_j = r_j                                           if episode terminates at j+1
        y_j = r_j + gamma * max_{a'} Q_hat(phi_{j+1}, a'; theta-)   otherwise
    and returns (y_j - Q(phi_j, a_j; theta))^2 averaged over the batch (the paper
    additionally clips the error term to [-1, 1] -- omitted here, plain MSE is used).
    """
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    # Mixed-precision training
    with torch.autocast(device_type=device.type):
        # Q(phi_j, a_j; theta) for the action actually taken
        state_action_values = net(states_t).gather(
            1, actions_t.unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            # max_{a'} Q_hat(phi_{j+1}, a'; theta-) using the target network
            next_state_values = tgt_net(new_states_t).max(1)[0]
            # terminal-state guard: y_j = r_j when episode ended at j+1
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
        lambda: wrappers.make_env(
            gym.make("ALE/Pong-v5", 
                     frameskip=1, 
                     repeat_action_probability=0.0))
        for _ in range(N_ENVS)
    ]
    env = gym.vector.AsyncVectorEnv(env_factories)
    assert isinstance(env.single_observation_space, gym.spaces.Box)
    assert isinstance(env.single_action_space, gym.spaces.Discrete)

    # Algorithm 1: "Initialize action-value function Q with random weights theta"
    raw_net = dqn_model.DQN(
        env.single_observation_space.shape, 
        env.single_action_space.n).to(device)
    
    net = cast(dqn_model.DQN, torch.compile(raw_net, backend="cudagraphs"))
    
    # Algorithm 1: "Initialize target action-value function Q_hat 
    # with weights theta- = theta"
    tgt_net = cast(dqn_model.DQN, torch.compile(
        dqn_model.DQN(
            env.single_observation_space.shape, 
            env.single_action_space.n).to(device),
        backend="cudagraphs"))
    
    writer = SummaryWriter(comment="-pong-dqn")
    
    print(net)
    print(f"Actions: {env.single_action_space.n}")

    # Algorithm 1: "Initialize replay memory D to capacity N"
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # Mixed-precision training
    scaler = GradScaler("cuda")
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_ts = ts
    best_m_reward = None
    solved = False
    speed = 0.0

    # Algorithm 1 outer loop ("For episode = 1, M do" merged with "For t = 1, T do":
    # vectorised envs auto-reset, so a single loop covers both).
    while not solved:
        frame_idx += N_ENVS
        # Linear epsilon annealing from EPSILON_START to EPSILON_FINAL
        epsilon = max(
            EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        # Inner-loop steps: select a_t (eps-greedy), execute, store transition in D
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

        # Wait until D holds enough transitions before sampling minibatches
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # Algorithm 1: "Sample random minibatch of transitions: 
        # (phi_j, a_j, r_j, phi_{j+1}) from D"
        # then "Perform a gradient descent step on: 
        # (y_j - Q(phi_j, a_j; theta))^2 w.r.t. theta"
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        scaler.scale(loss_t).backward()
        scaler.step(optimizer)
        scaler.update()

        # Algorithm 1: "Every C steps reset Q_hat = Q".
        # Here replaced by Polyak averaging (soft update): 
        # theta- <- (1 - TAU) * theta- + TAU * theta.
        with torch.no_grad():
            for p, p_tgt in zip(net.parameters(), tgt_net.parameters()):
                p_tgt.data.mul_(1 - TAU).add_(TAU * p.data)
    writer.close()
