"""A2C agent: n-step transition tracker, vectorised-env interaction, and batch packing.

Holds the on-policy data-collection machinery shared by `a2c_pong.py`:

  - `Experience` -- one n-step transition (s_t, a_t, R_t^{(n)}, s_{t+n}).
  - `NStepTracker` -- per-env sliding window that emits Experiences with
    discounted n-step reward sums (Mnih et al. 2016, Algorithm S3 inner
    "for i in {t-1, ..., t_start} do  R <- r_i + gamma R" loop).
  - `Agent` -- samples actions a_t ~ pi(.|s_t; theta) for all NUM_ENVS envs
    in lock-step (the synchronous A2C design from
    https://openai.com/index/openai-baselines-acktr-a2c/, where workers wait
    for each other and the policy update is a single GPU batch instead of
    A3C's asynchronous per-thread updates).
  - `batch_to_tensors` -- builds (states, actions, ref_values) tensors with
    the n-step bootstrap target  R = r_t + gamma r_{t+1} + ... + gamma^k V(s_{t+k}; theta_v)
    for non-terminal tails (R = sum of rewards only when s_{t+k} is terminal).
"""
import collections
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from core.model import AtariA2C


@dataclass
class Experience:
    """A single n-step transition: first state, action taken, discounted
    reward sum over n steps, and the state n steps later (None if terminal).

    Corresponds to one (s_i, a_i, R_i, s_{i+n}) tuple from the inner loop of
    Algorithm S3 in Mnih et al. 2016 (Asynchronous Methods for Deep RL).
    """
    state: np.ndarray
    action: int
    reward: float
    new_state: np.ndarray | None


class NStepTracker:
    """Accumulates per-environment transitions and emits n-step experiences.

    Maintains a sliding window of (state, action, reward) tuples per env.
    When the window is full, it collapses the entries into a single
    Experience with a discounted reward sum  R = sum_{i=0..n-1} gamma^i r_{t+i}
    (Algorithm S3 inner loop). On episode boundaries the window is flushed
    so no transition crosses episode limits, and the tail experiences carry
    new_state=None so `batch_to_tensors` skips the V(s') bootstrap (R = 0
    branch in the paper).
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
            # episode over -- flush everything with no bootstrap state
            while buf:
                results.append(self._pack(buf, new_state=None))
                buf.popleft()
        elif len(buf) == self.n_steps:
            results.append(self._pack(buf, new_state=new_state))
            buf.popleft()

        return results


class Agent:
    """Interacts with vectorised environments using the current policy.

    Synchronous A2C data-collection step: every env advances by one frame
    using a_t ~ pi(.|s_t; theta), then all transitions are pushed into the
    n-step tracker. Unlike A3C's per-thread asynchronous updates
    (Algorithm S3), the OpenAI A2C variant lets all workers contribute to a
    single batched gradient step on GPU
    (https://openai.com/index/openai-baselines-acktr-a2c/).
    """

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
        """Advance all envs by one step; return new experiences 
        and any finished episode rewards."""
        states_t = torch.as_tensor(self.states, dtype=torch.float32).to(self.device)
        logits_t, _ = self.net(states_t)
        probs = F.softmax(logits_t, dim=1).cpu().numpy()

        # Algorithm S3: "Perform a_t according to policy pi(a_t | s_t; theta')"
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

    Implements the n-step return target from Algorithm S3:
        R = 0                          if s_{t+k} is terminal
        R = V(s_{t+k}; theta_v)        otherwise (bootstrap)
        R <- r_i + gamma * R           for i in {t-1, ..., t_start}
    Here `exp.reward` already holds  sum_{i=0..k-1} gamma^i r_{t+i}, so the
    bootstrap reduces to adding  gamma^k * V(s_{t+k}; theta_v)  for
    non-terminal tails only.
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
