"""Perception extraction backbone using pretrained ResNet-FPN.

This module provides PerceptionExtraction which builds a Feature Pyramid
Network (FPN) style backbone commonly used by detectors like Faster R-CNN.
It uses torchvision's helper to construct a ResNet backbone with FPN and
keeps parameters frozen (no training).

Usage:
    from src.models.backbone.perception_extraction import PerceptionExtraction
    model = PerceptionExtraction(backbone_name='resnet50', pretrained=True)
    features = model(images)

The forward returns either a dict of feature maps (like torchvision detectors)
or a single tensor depending on implementation; here we return the OrderedDict
coming from the backbone to be compatible with detection heads.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from collections import OrderedDict

import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork


class PerceptionExtraction(nn.Module):
    """ResNet-FPN backbone wrapper with optional removal of res1.

    Special behavior for when the input is already the res2 feature map
    (i.e. 256 channels). In that case set remove_res1=True and the module
    will skip the ResNet stem and layer1, treating the module input as c2
    (res2). The rest of the ResNet layers (layer2..layer4) and the FPN are
    left unchanged.

    Parameters:
        backbone_name: name of torchvision resnet backbone (only resnet50
            is guaranteed/tested here).
        pretrained: whether to load pretrained weights for the ResNet
        in_channels: channels of the input tensor. If remove_res1 is True,
            this should match the expected res2 channels (256 for resnet50).
        out_channels: number of channels produced by the FPN feature maps.
        remove_res1: if True, skip the ResNet stem and layer1 and treat the
            input as res2 (c2). If False, builds a normal ResNet-FPN.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 256,
        out_channels: int = 256,
        remove_res1: bool = True,
        load_detection_pretrained: bool = True,
        freeze_fpn: bool = True,
    ):
        super().__init__()

        # Only implement the remove_res1 shortcut for resnet50; fallback to
        # torchvision's standard resnet_fpn behavior if not requested.
        self.remove_res1 = remove_res1
        self.backbone_name = backbone_name

        if remove_res1:
            # load resnet50 and keep layers 2..4
            try:
                resnet = getattr(models, backbone_name)(pretrained=pretrained)
            except Exception as e:
                raise RuntimeError(f"Failed to construct {backbone_name}: {e}")

            # layer2..layer4 compute c3,c4,c5 when given c2 as input
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

            # For ResNet50 the channel sizes are known: c2=256, c3=512,
            # c4=1024, c5=2048. We expect input to have c2 channels.
            in_channels_list = [in_channels, 512, 1024, 2048]


            # build a simple FPN that consumes [c2,c3,c4,c5] and produces
            # out_channels for each pyramid level
            self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)

            # initialize FPN weights similar to torchvision (Kaiming)
            self._init_fpn_weights()

            # a small pooling to create p6 from the last output
            self.pool = nn.MaxPool2d(kernel_size=1, stride=2)

            # freeze resnet parameters (we don't want to train these)
            for p in self.layer2.parameters():
                p.requires_grad = False
            for p in self.layer3.parameters():
                p.requires_grad = False
            for p in self.layer4.parameters():
                p.requires_grad = False

            # optionally load pretrained weights for the full detection model
            # (this may download weights for Faster R-CNN backbone+FPN)
            if load_detection_pretrained:
                try:
                    from torchvision.models import detection as detection_models

                    det = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
                    det_sd = det.state_dict()
                    my_sd = self.state_dict()

                    # copy any tensor from det_sd whose key (without leading
                    # 'backbone.') matches the suffix of one of our keys.
                    mapped = 0
                    for k, v in det_sd.items():
                        if not k.startswith("backbone."):
                            continue
                        sub = k[len("backbone.") :]
                        # find a key in our state_dict that endswith sub
                        for my_k in list(my_sd.keys()):
                            if my_k.endswith(sub):
                                my_sd[my_k] = v
                                mapped += 1
                                break

                    # load mapped params (non-strict to allow missing keys)
                    self.load_state_dict(my_sd, strict=False)
                except Exception:
                    # if download or mapping fails, keep silent and continue
                    # with pretrained ResNet and randomly init FPN
                    pass

            if freeze_fpn:
                for p in self.fpn.parameters():
                    p.requires_grad = False

        else:
            # fallback: build the standard torchvision ResNet-FPN via
            # IntermediateLayerGetter + FeaturePyramidNetwork. This branch is
            # left minimal because the repository originally used
            # resnet_fpn_backbone(). If needed later we can mirror that
            # implementation here.
            try:
                resnet = getattr(models, backbone_name)(pretrained=pretrained)
            except Exception as e:
                raise RuntimeError(f"Failed to construct {backbone_name}: {e}")

            # extract layer1..layer4 outputs (named layer1..layer4 in resnet)
            return_layers = {"layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5"}
            self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)

            in_channels_list = [256, 512, 1024, 2048]
            self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)
            self._init_fpn_weights()
            self.pool = nn.MaxPool2d(kernel_size=1, stride=2)

            for p in self.body.parameters():
                p.requires_grad = False

            if load_detection_pretrained:
                try:
                    from torchvision.models import detection as detection_models

                    det = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
                    det_sd = det.state_dict()
                    my_sd = self.state_dict()
                    mapped = 0
                    for k, v in det_sd.items():
                        if not k.startswith("backbone."):
                            continue
                        sub = k[len("backbone.") :]
                        for my_k in list(my_sd.keys()):
                            if my_k.endswith(sub):
                                my_sd[my_k] = v
                                mapped += 1
                                break
                    self.load_state_dict(my_sd, strict=False)
                except Exception:
                    pass

            if freeze_fpn:
                for p in self.fpn.parameters():
                    p.requires_grad = False

    def _init_fpn_weights(self):
        """Initialize conv layers in the FPN with Kaiming normal, biases zeroed."""
        for m in self.fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the input through the modified backbone.

        When remove_res1 is True the input x is expected to be the res2
        feature map (B, 256, H, W). The module will compute res3..res5 by
        passing through resnet.layer2..layer4 and then construct the FPN.

        Returns an OrderedDict of 5 feature maps keyed by 'p2'..'p6'.
        """
        if self.remove_res1:
            # treat x as c2 (res2)
            c2 = x
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)

            features = OrderedDict()
            # keys correspond to increasing resolution levels
            features["c2"] = c2
            features["c3"] = c3
            features["c4"] = c4
            features["c5"] = c5

            # FeaturePyramidNetwork expects a dict[str, Tensor]
            fpn_out = self.fpn(features)

            # fpn_out will contain keys matching the input keys; convert to
            # p2..p5 naming and append p6 produced by pooling p5
            out = OrderedDict()
            out["p2"] = fpn_out["c2"]
            out["p3"] = fpn_out["c3"]
            out["p4"] = fpn_out["c4"]
            out["p5"] = fpn_out["c5"]
            # produce p6 by downsampling p5
            out["p6"] = self.pool(out["p5"]) if "p5" in out else self.pool(c5)

            return out

        else:
            # standard branch: pass through body then FPN and add p6
            features = self.body(x)
            fpn_out = self.fpn(features)
            out = OrderedDict()
            # map c2..c5 to p2..p5
            out["p2"] = fpn_out["c2"]
            out["p3"] = fpn_out["c3"]
            out["p4"] = fpn_out["c4"]
            out["p5"] = fpn_out["c5"]
            out["p6"] = self.pool(out["p5"]) if "p5" in out else None
            return out

