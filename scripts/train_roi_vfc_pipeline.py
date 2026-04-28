"""Minimal training pipeline for ROI_VFC.

This script trains the model on synthetic random frame pairs to demonstrate the
training flow and how to call the model's compress/decompress helpers. Adapt
the dataset parts to use your real dataset loaders.
"""
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.roi_vfc import ROI_VFC


class RandomPairDataset(Dataset):
    """Yields random image pairs (cur, ref) in [0,1].

    Replace this with a dataset that yields real image tensors of shape
    (C, H, W) when training on real data.
    """
    def __init__(self, length=1000, size=(3, 128, 128)):
        self.length = length
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        c, h, w = self.size
        cur = torch.rand(c, h, w)
        ref = torch.rand(c, h, w)
        return cur, ref


def collate_fn(batch):
    cur = torch.stack([b[0] for b in batch], dim=0)
    ref = torch.stack([b[1] for b in batch], dim=0)
    return cur, ref


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = ROI_VFC()
    model = model.to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    dataset = RandomPairDataset(length=1000, size=(3, args.height, args.width))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        running_loss = 0.0
        for i, (x_cur, x_ref) in enumerate(loader):
            x_cur = x_cur.to(device)
            x_ref = x_ref.to(device)

            # features are extracted without_grad by design in ROI_VFC.extract_feature
            F_cur = model.extract_feature(x_cur)

            # forward expects reduced-channel feature; in the file the forward
            # signature takes F_cur directly (see model implementation)
            out = model(F_cur)
            x_hat = out.get("x_hat")
            rate = out.get("rate", 0.0)

            # compute pixel count for bpp
            B = x_cur.size(0)
            H = x_cur.size(-2)
            W = x_cur.size(-1)
            pixels = float(B * H * W)

            # MSE between original feature and reconstructed feature
            # the forward returns reconstructed feature in the same shape as F_cur
            if x_hat is None:
                recon_loss = torch.tensor(0.0, device=device)
            else:
                # ensure same dtype/device
                recon_loss = F_cur.to(x_hat.dtype)
                recon_loss = torch.nn.functional.mse_loss(x_hat, recon_loss)

            bpp = float(rate) / pixels if pixels > 0 else 0.0

            # combined RD loss: MSE + lambda * bpp
            loss = recon_loss + model.lambda_rd * bpp

            optimizer.zero_grad()
            # recon_loss is tensor; bpp is scalar float -> convert to tensor
            total_loss = recon_loss + model.lambda_rd * torch.tensor(bpp, device=device)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if (i + 1) % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(f"Epoch {epoch+1} step {i+1}/{len(loader)} - avg loss: {avg:.6f} - bpp: {bpp:.6f}")
                running_loss = 0.0

        print(f"Epoch {epoch+1} finished in {time.time()-t0:.1f}s")

        # evaluation / sanity check: compress + decompress a batch
        model.eval()
        with torch.no_grad():
            # ensure coder CDFs are prepared before compress/decompress
            model.update(force=True)
            x_cur, x_ref = next(iter(loader))
            x_cur = x_cur.to(device)
            x_ref = x_ref.to(device)

            comp = model.compress_bytes(x_cur, x_ref=x_ref)
            string = comp.get("string")
            shape = comp.get("shape")
            bit = comp.get("bit")
            bpp_enc = comp.get("bpp")
            print(f"Compressed example -> bit: {bit}, bpp: {bpp_enc:.6f}")

            dec = model.decompress_bytes(x_ref, None, string, shape)
            x_hat_dec = dec.get("x_hat")
            if x_hat_dec is not None:
                # compute an MSE between forward-decoded and stream-decoded reconstructions
                # Note: shapes should match
                print("Decompressed example shape:", x_hat_dec.shape)

        model.train()

        # save checkpoint
        ckpt = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(ckpt, args.ckpt_path)
        print(f"Saved checkpoint to {args.ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--ckpt-path", type=str, default="roi_vfc_ckpt.pth")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
