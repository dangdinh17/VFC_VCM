import os
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.single_dataset import Single_Dataset
from src.models.roi_vfc import ROI_VFC


def parse_args():
    parser = argparse.ArgumentParser("Train ROI_VFC (DCVC-TCM style)")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda-rd', type=float, default=0.01)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--dataset-length', type=int, default=200)
    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'roi_vfc_epoch{epoch}.pth')
    torch.save({'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict()}, path)


def train():
    args = parse_args()
    device = torch.device(args.device)

    # dataset: dummy pairs (x_t, x_ref)
    ds = Single_Dataset(length=args.dataset_length, shape=(1, 3, 64, 64))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ROI_VFC(img_channels=3, feat_channels=64, y_chan=96, lambda_rd=args.lambda_rd)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_rate = 0.0
        for i, (x_t, x_ref) in enumerate(dl):
            # x_t, x_ref are (N, C, H, W)
            x_t = x_t.to(device)
            x_ref = x_ref.to(device)

            # Forward: ROI_VFC.forward currently uses the current frame as ref
            out = model(x_t)

            # get reconstructed feature and rate
            F_cur_hat = out['x_hat']
            rate = out['rate']
            aux_loss = out.get('aux_loss', 0.0)

            # compute target features (ground-truth) with model's feature extractor
            with torch.no_grad():
                F_cur = model.feat_extraction(x_t)

            # feature-level MSE
            mse = F.mse_loss(F_cur_hat, F_cur)

            # rate may be a python float; wrap as tensor on device
            if isinstance(rate, float):
                rate_t = torch.tensor(rate, device=device)
            else:
                rate_t = rate.to(device)

            if isinstance(aux_loss, float):
                aux_t = torch.tensor(aux_loss, device=device)
            else:
                aux_t = aux_loss.to(device)

            loss = mse + args.lambda_rd * rate_t + aux_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_mse += float(mse.item())
            epoch_rate += float(rate_t.item())
            global_step += 1

            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch} step {i+1}/{len(dl)} loss={loss.item():.4f} mse={mse.item():.4f} rate={rate_t.item():.2f}")

        # end epoch
        t1 = time.time()
        print(f"Epoch {epoch} finished in {t1-t0:.1f}s avg_loss={epoch_loss/len(dl):.4f} avg_mse={epoch_mse/len(dl):.4f} avg_rate={epoch_rate/len(dl):.4f}")

        # save checkpoint
        save_checkpoint(model, optimizer, epoch, args.save_dir)

    # After training, prepare entropy coder CDFs for evaluation / compression
    print("Running final update() to prepare entropy coder CDFs...")
    model.update(force=True)
    save_checkpoint(model, optimizer, 'final', args.save_dir)


if __name__ == '__main__':
    train()
