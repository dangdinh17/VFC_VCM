"""ROI-guided conditional decoder.

This module contains the `ROI_Guided_Conditional_Decoder` class extracted
from the previous combined implementation.
"""

from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn
from src.models.layers import *
from compressai.layers import *


class ROI_Guided_Conditional_Decoder(nn.Module):
    """Decoder that reconstructs main feature from bottleneck and pyramid.

    Args:
        in_ch_y: channels of the bottleneck y.
        out_ch: channels of reconstructed feature (should match main feature channels).
        num_levels: number of pyramid levels expected (default: 5). We assume
            each pyramid level has the same channel count as `out_ch`.
    """

    def __init__(self, feat_chan=64, out_chan=96):
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
            Up2xConv(3*feat_chan + out_chan, feat_chan),
            GDN(feat_chan, inverse=True),
            # ResidualBlock(feat_chan, feat_chan),
            VSSBlock(hidden_dim=feat_chan, ssm_d_state=2, ssm_conv=1)
        )
        self.second_forward = nn.Sequential(
            Up2xConv(2*feat_chan, feat_chan),
            GDN(feat_chan, inverse=True),
            VSSBlock(hidden_dim=feat_chan, ssm_d_state=2, ssm_conv=1),
        )
        self.third_forward = nn.Sequential(
            Conv(2*feat_chan, feat_chan),
            GDN(feat_chan, inverse=True),
            VSSBlock(hidden_dim=feat_chan, ssm_d_state=2, ssm_conv=1),
        )
        self.fourth_forward = nn.Sequential(
            Conv(2*feat_chan, feat_chan),
            GDN(feat_chan, inverse=True),
            VSSBlock(hidden_dim=feat_chan, ssm_d_state=2, ssm_conv=1),
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
    
    def forward(self, y_hat: torch.Tensor, f_tilde: Optional[torch.Tensor] = None, dec_pyr:Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Reconstruct main feature from y_hat and pyramid_feats.

        y_hat: [B, in_ch_y, H, W] (assumed to be at main resolution)
        pyramid_feats: either a list/tuple of features or a dict/OrderedDict from
        an FPN backbone. Dict inputs are converted to a list preserving order.
        """
        p2, p3, p4, p5, p6 = self.piramid_process(dec_pyr)
        fuse = self.fuse(p4, p5, p6)
        first_out = self.first_forward(torch.cat([y_hat, fuse], dim=1))
        second_out = self.second_forward(torch.cat([first_out, p3], dim=1))
        third_out = self.third_forward(torch.cat([second_out, p2], dim=1))
        f_hat = self.fourth_forward(torch.cat([third_out, f_tilde], dim=1))
        
        return f_hat

__all__ = ["ROI_Guided_Conditional_Decoder"]
