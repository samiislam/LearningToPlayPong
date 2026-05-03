"""DQN convolutional Q-network architecture from Mnih et al. 2015.

Defines the action-value approximator used by both training (`dqn_pong.py`) and
evaluation (`dqn_play.py`). The network maps a stacked 84x84x4 preprocessed
frame tensor to one Q-value per discrete action via three convolutional layers
(32@8x8/s4, 64@4x4/s2, 64@3x3/s1) followed by a 512-unit fully-connected layer.
Pixel scaling (/255) happens on-device inside `forward` to keep the replay
buffer in uint8 and minimise host-to-GPU bandwidth.
"""
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

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
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor):
        # scale on GPU
        xx = x / 255.0
        return self.fc(self.conv(xx))
