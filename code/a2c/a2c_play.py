#!/usr/bin/env python3
"""Greedy evaluation of a trained A2C checkpoint on Pong.

Loads weights produced by `a2c_pong.py` (saved as `a2c-model-best.dat`
whenever a new best 100-episode mean reward is reached) into a fresh
`AtariA2C`, plays a single Pong episode by taking the argmax over the
policy-head logits (no sampling, no entropy exploration), records the
rendered frames to disk via `gym.wrappers.RecordVideo`, and prints the
final score plus a histogram of action selections.
"""
import gymnasium as gym
import argparse
import numpy as np
import collections

import torch

from core import model
from core import wrappers

AtariA2C = model.AtariA2C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-r", "--record", required=True, help="Directory for video")
    args = parser.parse_args()

    env = wrappers.make_env(gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0,
                                     render_mode="rgb_array"))
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    net = AtariA2C(env.observation_space.shape, int(env.action_space.n))
    state = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c: collections.Counter[int] = collections.Counter()

    while True:
        state_v = torch.tensor(np.expand_dims(state, 0))
        logits, _ = net(state_v)
        action = int(logits.argmax(dim=1).item())
        c[action] += 1
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += float(reward)
        if is_done or is_trunc:
            break
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()
