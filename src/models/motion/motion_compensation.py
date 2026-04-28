import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import *


class MotionCompensation(nn.Module):
    """Motion compensation module.

    Inputs:
      - F_ref: reference feature map (B, C, H, W)
      - m_t: motion feature produced by the MotionEstimator

    The module applies a DepthConv to the reference, processes via Mamba
    blocks, then uses modulatory maps derived from `m_t` (gating and add)
    to adaptively combine features. Finally it outputs the compensated
    feature map with the same channel count as `F_ref`.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.conv = Conv(channels, channels)
        self.mamba1 = VSSBlock(hidden_dim=channels, ssm_d_state=2, ssm_conv=1)
        
        self.mamba1_down = nn.Sequential(
            Down2xConv(channels, channels),
            VSSBlock(hidden_dim=channels, ssm_d_state=2, ssm_conv=1)
        )
        
        self.down = Down2xConv(channels, channels) 


        self.mamba2_up = nn.Sequential(
            Up2xConv(channels, channels),
            VSSBlock(hidden_dim=channels, ssm_d_state=2, ssm_conv=1)
        )
        self.mamba2 = VSSBlock(hidden_dim=channels, ssm_d_state=2, ssm_conv=1)
        

    def forward(self, f_ref: torch.Tensor, m_cur_hat: torch.Tensor) -> torch.Tensor:
        conv_ref = self.conv(f_ref)
        mamba1_out = self.mamba1(conv_ref)
        mamba1_down_out = self.mamba1_down(mamba1_out)
        down_out = self.down(m_cur_hat)
        mamba2_up_out = self.mamba2_up(down_out * mamba1_down_out) + mamba1_out
        f_cur_hat = self.mamba2(mamba2_up_out * m_cur_hat)
        return f_cur_hat


if __name__ == "__main__":
    B, C, H, W = 2, 16, 64, 64
    F_ref = torch.randn(B, C, H, W)
    m_t = torch.randn(B, C, H, W)
    mc = MotionCompensation(channels=C)
    out = mc(F_ref, m_t)
    print(out.shape)
