import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, dataset, optimizer, loss_fn, device="cpu", batch_size=1):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, iterations=2):
        it = 0
        self.model.train()
        while it < iterations:
            for x_t, x_ref in self.dataloader:
                # ensure shape [B, C, H, W]
                if x_t.dim() == 3:
                    x_t = x_t.unsqueeze(0)
                    x_ref = x_ref.unsqueeze(0)

                x_t = x_t.to(self.device)
                x_ref = x_ref.to(self.device)

                self.optimizer.zero_grad()
                x_hat, rate = self.model(x_t, x_ref)
                loss, d, r = self.loss_fn(x_hat, x_t, rate)
                loss.backward()
                self.optimizer.step()

                print(f"iter={it} loss={loss.item():.6f} distortion={d.item():.6f} rate={r.item():.6f}")
                it += 1
                if it >= iterations:
                    return
