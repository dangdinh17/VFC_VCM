from typing import Optional

import torch
import torch.nn as nn
from src.models.layers import *
__all__ = ["MotionEncoder", "HyperiorEntropyEncoder"]


class MotionEncoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            Down2xConv(in_channels, in_channels),
            nn.ReLU(inplace=True),
            Down2xConv(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, m_cur: torch.Tensor) -> torch.Tensor:
        return self.net(m_cur)
    
class HyperiorEntropyEncoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            Down2xConv(in_channels, in_channels),
            nn.ReLU(inplace=True),
            Down2xConv(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, m_cur: torch.Tensor) -> torch.Tensor:
        return self.net(m_cur)