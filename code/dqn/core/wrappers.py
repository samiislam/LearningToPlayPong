"""Gymnasium observation wrappers for ALE/Atari environments.

Each class wraps a `gym.Env` (typically the raw `ALE/Pong-v5` env) and adapts
its observation stream so the DQN can consume it:

  - `ImageToPyTorch` - reorders observation axes from HWC (Atari/OpenCV layout)
    to CHW (the layout `torch.nn.Conv2d` expects).
  - `BufferWrapper` - stacks the last `n_steps` preprocessed frames into a
    single observation, giving the network temporal context (the m=4 stack
    from Mnih et al. 2015).
  - `make_env` - composes Gymnasium's `AtariPreprocessing` (frame max-pool,
    grayscale, 84x84 resize, terminal-on-life-loss) with the two wrappers
    above to produce the full 84x84x4 input pipeline used during training
    and evaluation.
"""
import collections

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import AtariPreprocessing

gym.register_envs(ale_py)


class ImageToPyTorch(gym.ObservationWrapper):
    """Convert observations from HWC to CHW layout expected by PyTorch."""

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(),
            shape=new_shape, dtype=obs.dtype.type)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    """Stack the last *n_steps* frames into a single observation."""

    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0),
            obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype.type)
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(self, *, seed: int | None = None,
              options: dict | None = None):
        assert self.buffer.maxlen is not None
        for _ in range(self.buffer.maxlen - 1):
            obs_space = self.env.observation_space
            assert isinstance(obs_space, spaces.Box)
            self.buffer.append(obs_space.low)
        obs, extra = self.env.reset(seed=seed, options=options)
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)


def make_env(env):
    """Apply the preprocessing map phi from Mnih et al. 2015 (Methods, "Preprocessing").

    AtariPreprocessing performs: per-pixel max over the current and previous frame
    (removes Atari sprite flicker), Y-channel luminance extraction, and rescaling
    to 84x84. BufferWrapper then stacks the m=4 most recent preprocessed frames
    into the network input, matching the 84x84x4 tensor described in the paper.
    """
    env = AtariPreprocessing(env, noop_max=0, 
                             frame_skip=4, screen_size=(84,84),
                             terminal_on_life_loss=True,
                             grayscale_obs=True, grayscale_newaxis=True, 
                             scale_obs=False)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env
