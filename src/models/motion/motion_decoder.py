"""Motion decoder module.

Provides `MotionDecoder` which upsamples a latent motion tensor back to
an estimated motion/flow tensor using transpose convolutions.

This is a minimal, self-contained module intended as a placeholder
— replace with a proper decoder nn.Module when integrating into the
full model.
"""
from typing import Optional

import torch
import torch.nn as nn
from src.models.layers import *

class MotionDecoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Up2xConv(out_channels, in_channels),
            nn.ReLU(inplace=True),
            Up2xConv(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)