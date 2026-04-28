import torch
import torch.nn as nn


class DistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x_hat, x):
        return self.mse(x_hat, x)
