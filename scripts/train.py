import os
import sys
import torch
import torch.optim as optim

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config import load_config
from src.models.roi_vfc import TransVFCModel
from src.datasets.single_dataset import Single_Dataset
from src.losses.rate_distortion import RateDistortionLoss
from src.trainers.trainer import Trainer


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "configs", "train.yaml")
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransVFCModel(img_channels=3, feat_channels=16, lambda_rd=cfg.get("lambda", 0.01))
    dataset = Single_Dataset(length=10, shape=(1, 3, 64, 64))
    loss_fn = RateDistortionLoss(lambda_rd=cfg.get("lambda", 0.01))
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))
    trainer = Trainer(model, dataset, optimizer, loss_fn, device=device, batch_size=1)
    trainer.train(iterations=cfg.get("iterations", 2))


if __name__ == "__main__":
    main()
