"""Feature extraction backbone using a pretrained ResNet.

This module provides FeatureExtraction which wraps a torchvision ResNet
backbone (ResNet50 by default) pretrained on ImageNet. All parameters are
frozen (requires_grad=False) so it won't be trained further.

Usage:
    from src.models.backbone.feature_extraction import FeatureExtraction
    model = FeatureExtraction(pretrained=True, backbone_name='resnet50')
    feats = model(images)  # returns feature map from last conv block
"""
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtraction(nn.Module):
    """Simple wrapper around a pretrained ResNet to extract features.

    This extractor returns intermediate features from the ResNet backbone
    limited to the "res2" stage (torchvision's `layer1`). In other words,
    we only keep layers up to `resnet.layer1`. The backbone parameters are
    frozen so this module is suitable as a fixed feature extractor.
    """

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        # choose a ResNet architecture from torchvision
        if not hasattr(models, backbone_name):
            raise ValueError(f"Unknown backbone: {backbone_name}")
        resnet_constructor = getattr(models, backbone_name)
        # load pretrained model
        resnet = resnet_constructor(pretrained=pretrained)

        # take layers up to res2 (torchvision's layer1) and stop there.
        # torchvision ResNet modules layout: conv1, bn1, relu, maxpool,
        # layer1, layer2, layer3, layer4, avgpool, fc
        # We intentionally stop after `layer1` so the extractor outputs the
        # intermediate feature map corresponding to the res2 stage.
        modules = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        ]
        self.body = nn.Sequential(*modules)

        # freeze parameters (do not train)
        for p in self.body.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature map from the backbone.

        Args:
            x: input images, tensor of shape (B, C, H, W)

        Returns:
            feature map tensor, typically (B, C_out, H_out, W_out)
        """
        return self.body(x)

