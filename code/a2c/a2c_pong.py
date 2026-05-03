"""Advantage Actor-Critic (A2C) on Pong.

Implements the synchronous A2C variant of "Asynchronous Methods for Deep
Reinforcement Learning" (Mnih et al. 2016, ICML). Algorithm S3 of the paper
describes A3C: each actor-learner thread runs its own copy of the policy
pi(a_t | s_t; theta) and value function V(s_t; theta_v), unrolls tmax steps
of experience, then asynchronously applies its accumulated gradients to the
shared parameters.

A2C (https://openai.com/index/openai-baselines-acktr-a2c/) is the
*synchronous* variant: all NUM_ENVS workers step in lock-step, the batch is
concatenated across workers, and a single GPU forward/backward pass updates
the shared network. OpenAI found this "performs better than our
asynchronous implementations" because it "allows for larger batch sizes"
that use a GPU more efficiently than CPU-bound A3C threads.

Loss (Algorithm S3 inner loop):
    R = 0                          if terminal
    R = V(s_{t+k}; theta_v)        otherwise (bootstrap)
    for i in {t-1, ..., t_start}: R <- r_i + gamma R
    dtheta   += grad_theta log pi(a_i|s_i;theta) (R - V(s_i;theta_v))
    dtheta_v += grad_theta_v (R - V(s_i;theta_v))^2
plus the entropy bonus  beta * H(pi(.|s_t;theta))  ("we found that adding
the entropy of the policy pi to the objective function improved exploration
by discouraging premature convergence to suboptimal deterministic
policies"), Section 4.

Deviations from the paper:
  - Synchronous A2C, not asynchronous A3C: one GPU update per batch.
  - Adam (instead of shared-statistics RMSProp from the paper).
  - Vectorised gymnasium envs replace per-thread actor-learners.
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from core import model
from core import wrappers
from core.a2c_agent import Agent, Experience, batch_to_tensors

BATCH_SIZE = 128
CLIP_GRAD = 0.1
ENTROPY_BETA = 0.01        # beta in Section 4: weight of the H(pi) entropy bonus
GAMMA = 0.99
LEARNING_RATE = 0.001
MEAN_REWARD_BOUND = 19
NUM_ENVS = 50              # synchronous A2C workers (replaces A3C threads)
REWARD_STEPS = 4           # tmax in Algorithm S3 (n-step return horizon)


AtariA2C = model.AtariA2C


def calc_loss(net: AtariA2C, states_t: torch.Tensor, actions_t: torch.Tensor,
              vals_ref_t: torch.Tensor, entropy_beta: float = ENTROPY_BETA
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                         torch.Tensor, torch.Tensor]:
    """Compute the three A2C loss components (Algorithm S3 + entropy bonus).

    Returns (policy_loss, value_loss, entropy_loss, advantage, values) where:
      - policy_loss  = -mean[ (R - V(s)).detach() * log pi(a|s) ]   (actor)
      - value_loss   =  mean[ (R - V(s))^2 ]                        (critic)
      - entropy_loss = beta * mean[ sum_a pi(a|s) log pi(a|s) ]     (Section 4)
    The advantage is detached so the policy gradient only flows through
    log pi, matching the paper's separated dtheta / dtheta_v updates.
    """
    logits_t, value_t = net(states_t)
    loss_value_t = F.mse_loss(value_t.squeeze(-1), vals_ref_t)

    log_prob_t = F.log_softmax(logits_t, dim=1)
    # advantage A(s,a) = R - V(s); detached so grad flows only through log pi
    adv_t = vals_ref_t - value_t.squeeze(-1).detach()
    log_act_t = log_prob_t[range(len(states_t)), actions_t]
    log_prob_actions_t = adv_t * log_act_t
    loss_policy_t = -log_prob_actions_t.mean()

    # entropy bonus: beta * H(pi) (note: sign is + because sum pi log pi <= 0,
    # so subtracting this from total loss is equivalent to maximising entropy)
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
        lambda: wrappers.make_env(
            gym.make("ALE/Pong-v5", 
                     frameskip=1, 
                     repeat_action_probability=0.0))
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
    # Algorithm S3: shared policy/value parameters theta, theta_v
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

    # Outer "repeat ... until T > T_max" loop of Algorithm S3.
    # In synchronous A2C, NUM_ENVS workers contribute one transition each
    # per iteration; the batched gradient step replaces the asynchronous
    # per-thread updates of A3C.
    while not solved:
        # Inner unroll: one step in every env (Algorithm S3 inner repeat-until).
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

        # Algorithm S3: build the n-step targets R for every (s_i, a_i) in
        # the unroll, then "Perform asynchronous update of theta using dtheta
        # and of theta_v using dtheta_v" -- here a single synchronous
        # batched optimiser step on GPU (the A2C variant).
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
