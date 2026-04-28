import torch
import torch.nn as nn
from .distortion import DistortionLoss


class RateDistortionLoss(nn.Module):
    def __init__(self, lambda_rd=0.01):
        super().__init__()
        self.dist = DistortionLoss()
        self.lambda_rd = lambda_rd

    def forward(self, x_hat, x, rate):
        d = self.dist(x_hat, x)
        # rate can be tensor scalar
        if isinstance(rate, torch.Tensor):
            r = rate
        else:
            r = torch.tensor(float(rate), device=x.device)
        loss = d + self.lambda_rd * r
        return loss, d, r


if __name__ == "__main__":
    import torch
    rd = RateDistortionLoss(lambda_rd=0.1)
    x = torch.randn(2, 3, 64, 64)
    y = x + 0.1 * torch.randn_like(x)
    l, d, r = rd(y, x, torch.tensor(1.0))
    print(l, d, r)
