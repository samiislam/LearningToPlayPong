"""A2C network: shared convolutional trunk with policy (actor) and value (critic) heads.

Mirrors the convolutional encoder from Mnih et al. 2015 DQN
(32@8x8/s4, 64@4x4/s2, 64@3x3/s1) -- the same trunk used by Mnih et al. 2016
for the A3C/A2C Atari experiments. The trunk output feeds two independent
512-unit MLPs:

  - `policy` head -> n_actions logits, softmax-ed in `a2c_pong.py` to give
    pi(a_t | s_t; theta) for sampling and the log-prob in the policy gradient.
  - `value`  head -> a single scalar V(s_t; theta_v) used as the critic
    baseline / bootstrap in the n-step return target.

Pixel scaling (/255) happens on-device inside `forward` to keep upstream
buffers in uint8 and minimise host-to-GPU bandwidth.
"""
import torch
import torch.nn as nn


class AtariA2C(nn.Module):
    """Shared-trunk CNN with separate policy and value heads."""

    def __init__(self, input_shape: tuple[int, ...], n_actions: int):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.policy = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xx = x / 255
        conv_out = self.conv(xx)
        return self.policy(conv_out), self.value(conv_out)
