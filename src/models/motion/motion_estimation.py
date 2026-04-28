import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import *


class MotionEstimation(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = Conv(2*channels, channels)
        
        self.down1 = nn.Sequential(
            Down2xConv(channels, channels),
            VSSBlock(hidden_dim=channels, ssm_d_state=2, ssm_conv=1)
        )
        self.down2 = nn.Sequential(
            Down2xConv(channels, channels),
            VSSBlock(hidden_dim=channels, ssm_d_state=2, ssm_conv=1)
        )
        self.up1 = nn.Sequential(
            Up2xConv(channels, channels),
            VSSBlock(hidden_dim=channels, ssm_d_state=2, ssm_conv=1)
        )
        self.up2 = nn.Sequential(
            Up2xConv(channels, channels),
            VSSBlock(hidden_dim=channels, ssm_d_state=2, ssm_conv=1)
        )
    def forward(self, f_cur: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_cur, f_ref], dim=1)
        conv_out = self.conv(x)
        down1_out = self.down1(conv_out)
        down2_out = self.down2(down1_out)
        up1_out = self.up1(down2_out)
        out = self.up2(up1_out + down1_out)
        m_cur = out + conv_out
        return m_cur


if __name__ == "__main__":
    m = MotionEstimation(in_channels=32, base_channels=64, out_channels=16)
    a = torch.randn(2, 16, 64, 64)
    b = torch.randn(2, 16, 64, 64)
    out = m(a, b)
    print(out.shape)
