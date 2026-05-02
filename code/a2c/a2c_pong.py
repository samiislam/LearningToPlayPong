#!/usr/bin/env python3
"""Advantage Actor-Critic (A2C) on Pong.

On-policy algorithm that trains a shared-trunk network with separate
policy (actor) and value (critic) heads.  Uses N parallel environments
for variance reduction and n-step returns for faster credit assignment.
"""
import argparse
import collections
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from core import model
from core import wrappers

BATCH_SIZE = 128
CLIP_GRAD = 0.1
ENTROPY_BETA = 0.01        # weight for the entropy bonus in the loss
GAMMA = 0.99
LEARNING_RATE = 0.001
MEAN_REWARD_BOUND = 19
NUM_ENVS = 50              # parallel envs for on-policy variance reduction
REWARD_STEPS = 4           # n-step return horizon


@dataclass
class Experience:
    """A single n-step transition: first state, action taken, discounted
    reward sum over n steps, and the state n steps later (None if terminal)."""
    state: np.ndarray
    action: int
    reward: float
    new_state: np.ndarray | None


AtariA2C = model.AtariA2C


class NStepTracker:
    """Accumulates per-environment transitions and emits n-step experiences.

    Maintains a sliding window of (state, action, reward) tuples per env.
    When the window is full, it collapses the entries into a single
    Experience with a discounted reward sum.  On episode boundaries the
    window is flushed so no transition crosses episode limits.
    """

    def __init__(self, n_envs: int, gamma: float, n_steps: int):
        self.gamma = gamma
        self.n_steps = n_steps
        self.buffers: list[collections.deque] = [
            collections.deque(maxlen=n_steps) for _ in range(n_envs)
        ]

    def _pack(self, buf: collections.deque, new_state: np.ndarray | None) -> Experience:
        """Collapse buffered transitions into one n-step experience."""
        reward = 0.0
        for _, _, r in reversed(buf):
            reward = r + self.gamma * reward
        state, action, _ = buf[0]
        return Experience(state=state, action=action, reward=reward,
                                   new_state=new_state)

    def step(self, env_idx: int, state: np.ndarray, action: int,
             reward: float, done: bool, new_state: np.ndarray | None
             ) -> list[Experience]:
        """Record one transition; return completed n-step experiences (if any)."""
        buf = self.buffers[env_idx]
        buf.append((state, action, reward))
        results: list[Experience] = []

        if done:
            # episode over — flush everything with no bootstrap state
            while buf:
                results.append(self._pack(buf, new_state=None))
                buf.popleft()
        elif len(buf) == self.n_steps:
            results.append(self._pack(buf, new_state=new_state))
            buf.popleft()

        return results


class Agent:
    """Interacts with vectorized environments using the current policy."""

    def __init__(self, env: gym.vector.VectorEnv, net: AtariA2C,
                 device: torch.device, gamma: float, reward_steps: int):
        self.env = env
        self.net = net
        self.device = device
        self.n_envs = env.num_envs
        self.tracker = NStepTracker(self.n_envs, gamma, reward_steps)
        self.total_rewards = np.zeros(self.n_envs)
        self.states, _ = self.env.reset()

    @torch.no_grad()
    def play_step(self) -> tuple[list[Experience], list[float]]:
        """Advance all envs by one step; return new experiences and any finished episode rewards."""
        states_t = torch.as_tensor(self.states, dtype=torch.float32).to(self.device)
        logits_t, _ = self.net(states_t)
        probs = F.softmax(logits_t, dim=1).cpu().numpy()

        # sample actions from the policy distribution
        actions = np.array([
            np.random.choice(len(p), p=p) for p in probs
        ])

        new_states, rewards, is_done, is_tr, infos = self.env.step(actions)
        self.total_rewards += rewards

        experiences: list[Experience] = []
        completed_rewards: list[float] = []

        for i in range(self.n_envs):
            done = bool(is_done[i]) or bool(is_tr[i])
            # on termination the auto-reset overwrites new_states[i],
            # so use final_observation to get the true last state
            if done and "final_observation" in infos:
                next_obs = infos["final_observation"][i]
            else:
                next_obs = new_states[i]

            exps = self.tracker.step(
                env_idx=i, state=self.states[i], action=int(actions[i]),
                reward=float(rewards[i]), done=done, new_state=next_obs)
            experiences.extend(exps)

            if done:
                completed_rewards.append(float(self.total_rewards[i]))
                self.total_rewards[i] = 0.0

        self.states = new_states
        return experiences, completed_rewards


def batch_to_tensors(batch: list[Experience], net: AtariA2C,
                 device: torch.device, gamma: float, reward_steps: int):
    """Convert a batch of n-step experiences into training tensors.

    For non-terminal experiences the discounted n-step reward is
    bootstrapped with V(s') from the current value head.
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    new_states = []
    for idx, exp in enumerate(batch):
        states.append(np.asarray(exp.state))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.new_state is not None:
            not_done_idx.append(idx)
            new_states.append(np.asarray(exp.new_state))

    states_t = torch.FloatTensor(np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # bootstrap non-terminal rewards with discounted value estimate
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        new_states_t = torch.FloatTensor(np.asarray(new_states)).to(device)
        last_vals_t = net(new_states_t)[1]
        last_vals_np = last_vals_t.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma ** reward_steps
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_t = torch.FloatTensor(rewards_np).to(device)
    return states_t, actions_t, ref_vals_t


def calc_loss(net: AtariA2C, states_t: torch.Tensor, actions_t: torch.Tensor,
              vals_ref_t: torch.Tensor, entropy_beta: float = ENTROPY_BETA
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                         torch.Tensor, torch.Tensor]:
    """Compute the three A2C loss components.

    Returns (policy_loss, value_loss, entropy_loss, advantage, values).
    The policy loss uses the advantage (ref_values - V(s)) to weight
    log-probabilities.  The entropy bonus encourages exploration.
    """
    logits_t, value_t = net(states_t)
    loss_value_t = F.mse_loss(value_t.squeeze(-1), vals_ref_t)

    log_prob_t = F.log_softmax(logits_t, dim=1)
    adv_t = vals_ref_t - value_t.squeeze(-1).detach()
    log_act_t = log_prob_t[range(len(states_t)), actions_t]
    log_prob_actions_t = adv_t * log_act_t
    loss_policy_t = -log_prob_actions_t.mean()

    prob_t = F.softmax(logits_t, dim=1)
    entropy_loss_t = entropy_beta * (prob_t * log_prob_t).sum(dim=1).mean()

    return loss_policy_t, loss_value_t, entropy_loss_t, adv_t, value_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda", help="Device to use, default=cuda")
    parser.add_argument("--use-async", default=False, action='store_true',
                        help="Use async vector env (A3C mode)")
    args = parser.parse_args()
    device = torch.device(args.dev)

    env_factories = [
        lambda: wrappers.make_env(gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0))
        for _ in range(NUM_ENVS)
    ]
    if args.use_async:
        env = gym.vector.AsyncVectorEnv(env_factories)
    else:
        env = gym.vector.SyncVectorEnv(env_factories)
    writer = SummaryWriter(comment="-pong-a2c")

    obs_shape = env.single_observation_space.shape
    assert obs_shape is not None
    act_space = env.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete)
    net = AtariA2C(obs_shape, int(act_space.n)).to(device)
    print(net)
    print(f"Actions: {act_space.n}")

    agent = Agent(env, net, device, gamma=GAMMA, reward_steps=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch: list[Experience] = []
    total_rewards: list[float] = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_ts = ts
    best_m_reward = None
    speed = 0.0
    solved = False

    while not solved:
        exps, completed_rewards = agent.play_step()
        frame_idx += NUM_ENVS

        batch.extend(exps)

        # update speed estimate when episodes finish
        if completed_rewards:
            now = time.time()
            elapsed = now - ts
            if elapsed > 0:
                speed = (frame_idx - ts_frame) / elapsed
            ts_frame = frame_idx
            ts = now
        for reward in completed_rewards:
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_ts))
            print(f"{elapsed} {frame_idx}: done {len(total_rewards)} games, "
                  f"reward {m_reward:.3f}, speed {speed:.2f} f/s")
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), "a2c-model-best.dat")
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                solved = True
                break

        if len(batch) < BATCH_SIZE:
            continue

        states_t, actions_t, vals_ref_t = batch_to_tensors(
            batch, net, device=device, gamma=GAMMA, reward_steps=REWARD_STEPS)
        batch.clear()

        optimizer.zero_grad()
        loss_policy_t, loss_value_t, entropy_loss_t, adv_t, value_t = calc_loss(
            net, states_t, actions_t, vals_ref_t)

        # two-phase backward: first policy alone (to capture its gradient
        # norms for logging), then entropy + value on top
        loss_policy_t.backward(retain_graph=True)
        grads = np.concatenate([
            p.grad.data.cpu().numpy().flatten()
            for p in net.parameters() if p.grad is not None
        ])

        loss_v = entropy_loss_t + loss_value_t
        loss_v.backward()
        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        optimizer.step()
        loss_v += loss_policy_t  # total loss for logging only

        writer.add_scalar("advantage", adv_t.mean().item(), frame_idx)
        writer.add_scalar("values", value_t.mean().item(), frame_idx)
        writer.add_scalar("batch_rewards", vals_ref_t.mean().item(), frame_idx)
        writer.add_scalar("loss_entropy", entropy_loss_t.item(), frame_idx)
        writer.add_scalar("loss_policy", loss_policy_t.item(), frame_idx)
        writer.add_scalar("loss_value", loss_value_t.item(), frame_idx)
        writer.add_scalar("loss_total", loss_v.item(), frame_idx)
        writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), frame_idx)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), frame_idx)
        writer.add_scalar("grad_var", np.var(grads), frame_idx)

    writer.close()
