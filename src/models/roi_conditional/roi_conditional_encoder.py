"""ROI-guided conditional encoder.

This module contains the `ROI_Guided_Conditional_Encoder` class extracted
from the previous combined implementation.
"""

from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn
from src.models.layers import *
from compressai.layers import *

class ROI_Guided_Conditional_Encoder(nn.Module):
    """Encoder that fuses a main feature with a pyramid of features.

    Args:
        in_ch_main: channels of the main input feature (`f_t`).
        out_ch: number of channels for output bottleneck `y`.
        num_levels: number of pyramid levels expected (default: 5). We assume
            each pyramid level has the same channel count as `in_ch_main`.
    """

    def __init__(self, feat_chan: int, out_chan: int):
        super().__init__()
        # projector for main feature
        # self.main_proj = ConvReLU(feat_chan, out_ch)

        # # optional projector for an auxiliary main-like feature (f_tilde)
        # self.tilde_proj = ConvReLU(feat_chan, out_ch)

        # # per-level projectors to unify channel dims before fusion. We assume
        # # each pyramid level has channels == in_ch_main, so project in_ch_main->out_ch
        # self.level_projs = nn.ModuleList([ConvReLU(feat_chan, out_ch) for _ in range(num_levels)])

        # # final fusion conv to produce bottleneck y
        # # fusion expects: main + (optional f_tilde) + num_levels projected features
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(out_ch * (2 + num_levels), out_ch, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        self.first_forward = nn.Sequential(
            Conv(2*feat_chan, feat_chan),
            GDN(feat_chan),
            # ResidualBlock(feat_chan, feat_chan),
            VSSBlock(hidden_dim=feat_chan, ssm_d_state=2, ssm_conv=1)
        )
        self.second_forward = nn.Sequential(
            Down2xConv(2*feat_chan, feat_chan),
            GDN(feat_chan),
            VSSBlock(hidden_dim=feat_chan, ssm_d_state=2, ssm_conv=1),
        )
        self.third_forward = nn.Sequential(
            Down2xConv(2*feat_chan, feat_chan),
            GDN(feat_chan),
            VSSBlock(hidden_dim=feat_chan, ssm_d_state=2, ssm_conv=1),
        )
        self.fourth_forward = nn.Sequential(
            Conv(4*feat_chan, out_chan),
            GDN(out_chan),
            VSSBlock(hidden_dim=out_chan, ssm_d_state=2, ssm_conv=1),
        )
        
        self.fuse = MultiScaleFusion()
    @staticmethod
    def piramid_process(pyramid_feats: Union[List[torch.Tensor], Dict[str, torch.Tensor], tuple] = None):
        p2 = pyramid_feats["p2"]
        p3 = pyramid_feats["p3"]
        p4 = pyramid_feats["p4"]
        p5 = pyramid_feats["p5"]
        p6 = pyramid_feats["p6"]
        return p2, p3, p4, p5, p6

    def forward(self, f_cur: torch.Tensor, f_tilde: Optional[torch.Tensor] = None, enc_pyr: Union[List[torch.Tensor], Dict[str, torch.Tensor], tuple] = None) -> torch.Tensor:
        """Fuse main feature and pyramid_feats into a bottleneck y.

        Pyramid features are upsampled to the main spatial size by
        bilinear interpolation, then projected and concatenated.

        Accepts either:
          - list/tuple of tensors (ordered from highest to lowest resolution), or
          - dict/OrderedDict as returned by torchvision FPN backbones; in that
            case the dict's values are used in iteration order.
        """
        p2, p3, p4, p5, p6 = self.piramid_process(enc_pyr)
        
        first_out = self.first_forward(torch.cat([f_cur, f_tilde], dim=1))
        # print("first_out.shape:", first_out.shape)
        # print("p2.shape:", p2.shape)
        second_out = self.second_forward(torch.cat([first_out, p2], dim=1))
        third_out = self.third_forward(torch.cat([second_out, p3], dim=1))
        
        fuse = self.fuse(p4, p5, p6)
        y = self.fourth_forward(torch.cat([fuse, third_out], dim=1))
        
        return y

__all__ = ["ROI_Guided_Conditional_Encoder"]
