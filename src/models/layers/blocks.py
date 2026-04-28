"""Small reusable layer blocks.

This module provides small building blocks used across models. We expose
ConvReLU which is a thin conv -> ReLU projector used by ROI modules.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReLU(nn.Module):
	"""Conv -> ReLU block.

	Simple 2D convolution followed by an in-place ReLU. Kept deliberately
	minimal so it's easy to reuse in lightweight modules.
	"""

	def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

class Conv(nn.Module):
	"""Conv -> ReLU block.

	Simple 2D convolution followed by an in-place ReLU. Kept deliberately
	minimal so it's easy to reuse in lightweight modules.
	"""

	def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

class Down2xConv(nn.Module):
	"""Downsample by 2x using stride-2 conv -> ReLU."""

	def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 5, stride=2, padding=2),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
class Up2xConv(nn.Module):
	"""Upsample by 2x using ConvTranspose2d -> ReLU."""

	def __init__(self, in_ch: int, out_ch: int, kernel: int = 2, stride: int = 2, padding: int = 1):
		super().__init__()
		self.net = nn.Sequential(
			nn.ConvTranspose2d(in_ch, out_ch, 5, stride=2, padding=2, output_padding=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
	
 
class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3):
        H, W = x1.shape[-2:]

        x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x1, x2, x3], dim=1)
        return out
    
__all__ = ["ConvReLU", "Down2xConv", "Up2xConv", "Conv", "MultiScaleFusion"]