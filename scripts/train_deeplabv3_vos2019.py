from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


@dataclass
class VOSItem:
	image_path: Path
	mask_path: Path


class VOS2019BinarySegmentationDataset(Dataset):
	"""DAVIS/VOS2019 dataset for binary segmentation (bg vs fg)."""

	def __init__(
		self,
		split_dir: Path,
		image_size: Tuple[int, int] = (480, 854),
		max_samples: int | None = None,
		seed: int = 42,
	) -> None:
		super().__init__()
		self.split_dir = split_dir
		self.img_root = split_dir / "JPEGImages"
		self.mask_root = split_dir / "Annotations"

		if not self.img_root.exists() or not self.mask_root.exists():
			raise FileNotFoundError(
				f"Missing JPEGImages/Annotations under {split_dir}. "
				"Expected VOS2019 folder structure."
			)

		self.items = self._collect_items(max_samples=max_samples, seed=seed)
		if len(self.items) == 0:
			raise RuntimeError(f"No image/mask pairs found under {split_dir}")

		h, w = int(image_size[0]), int(image_size[1])
		self.image_tf = transforms.Compose(
			[
				transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)
		self.mask_resize = transforms.Resize(
			(h, w), interpolation=transforms.InterpolationMode.NEAREST
		)

	def _collect_items(self, max_samples: int | None, seed: int) -> List[VOSItem]:
		pairs: List[VOSItem] = []
		for video_dir in sorted(self.mask_root.iterdir()):
			if not video_dir.is_dir():
				continue
			for mask_path in sorted(video_dir.glob("*.png")):
				stem = mask_path.stem
				img_path = self.img_root / video_dir.name / f"{stem}.jpg"
				if img_path.exists():
					pairs.append(VOSItem(image_path=img_path, mask_path=mask_path))

		if max_samples is not None and max_samples < len(pairs):
			rng = random.Random(seed)
			rng.shuffle(pairs)
			pairs = pairs[: max(1, max_samples)]

		return pairs

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		item = self.items[index]
		image = Image.open(item.image_path).convert("RGB")
		mask = Image.open(item.mask_path).convert("L")

		image_t = self.image_tf(image)
		mask_resized = self.mask_resize(mask)
		mask_np = np.array(mask_resized, dtype=np.uint8)
		# VOS2019 uses object IDs in mask; collapse all IDs > 0 into foreground=1.
		target_np = (mask_np > 0).astype(np.int64)
		target = torch.from_numpy(target_np)
		return image_t, target


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def make_model(num_classes: int, pretrained_backbone: bool = True) -> nn.Module:
	backbone_weights = (
		ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
	)
	model = deeplabv3_resnet50(
		weights=None,
		weights_backbone=backbone_weights,
		num_classes=num_classes,
		aux_loss=True,
	)
	return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_pixels = 0
	total_inter = 0
	total_union = 0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		logits = model(images)["out"]
		loss = F.cross_entropy(logits, targets)
		total_loss += float(loss.item()) * images.size(0)

		pred = logits.argmax(dim=1)
		total_correct += int((pred == targets).sum().item())
		total_pixels += int(targets.numel())

		pred_fg = pred == 1
		gt_fg = targets == 1
		total_inter += int((pred_fg & gt_fg).sum().item())
		total_union += int((pred_fg | gt_fg).sum().item())

	avg_loss = total_loss / max(1, len(loader.dataset))
	pix_acc = total_correct / max(1, total_pixels)
	iou_fg = total_inter / max(1, total_union)
	return {"loss": avg_loss, "pixel_acc": pix_acc, "iou_fg": iou_fg}


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	aux_weight: float,
	print_freq: int,
) -> Dict[str, float]:
	model.train()
	total_loss = 0.0

	for step, (images, targets) in enumerate(loader, start=1):
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		out = model(images)
		main_loss = F.cross_entropy(out["out"], targets)
		aux_loss = F.cross_entropy(out["aux"], targets) if "aux" in out else 0.0

		if isinstance(aux_loss, torch.Tensor):
			loss = main_loss + aux_weight * aux_loss
		else:
			loss = main_loss

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		total_loss += float(loss.item()) * images.size(0)

		if step % print_freq == 0 or step == len(loader):
			print(
				f"[train] step {step:04d}/{len(loader):04d} "
				f"loss={loss.item():.4f} main={main_loss.item():.4f}"
			)

	avg_loss = total_loss / max(1, len(loader.dataset))
	return {"loss": avg_loss}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser("Train DeepLabV3-ResNet50 on VOS2019")
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("configs/vos2019/deeplabv3_resnet50.yaml"),
		help="Path to yaml config file",
	)
	parser.add_argument(
		"--data-root",
		type=Path,
		default=None,
		help="Path to dataset root containing train/ and valid/",
	)
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--batch-size", type=int, default=None)
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--weight-decay", type=float, default=None)
	parser.add_argument("--num-workers", type=int, default=None)
	parser.add_argument("--height", type=int, default=None)
	parser.add_argument("--width", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--print-freq", type=int, default=None)
	parser.add_argument("--aux-weight", type=float, default=None)
	parser.add_argument("--max-train-samples", type=int, default=None)
	parser.add_argument("--max-valid-samples", type=int, default=None)
	parser.add_argument(
		"--save-dir", type=Path, default=None
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
	)
	return parser.parse_args()


def _read_config(config_path: Path) -> Dict[str, Any]:
	if not config_path.is_absolute():
		config_path = PROJECT_ROOT / config_path
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	return load_config(str(config_path)) or {}


def _resolve_setting(cli_val: Any, cfg_val: Any, default_val: Any) -> Any:
	if cli_val is not None:
		return cli_val
	if cfg_val is not None:
		return cfg_val
	return default_val


def _resolve_device(device_name: str | None) -> torch.device:
	if device_name in (None, "auto"):
		device_name = "cuda" if torch.cuda.is_available() else "cpu"
	return torch.device(device_name)


def main() -> None:
	args = parse_args()
	cfg = _read_config(args.config)

	model_cfg = cfg.get("model", {})
	data_cfg = cfg.get("data", {})
	train_cfg = cfg.get("training", {})
	output_cfg = cfg.get("output", {})

	seed = int(_resolve_setting(args.seed, cfg.get("seed"), 42))
	seed_everything(seed)

	data_root = Path(
		_resolve_setting(args.data_root, data_cfg.get("data_root"), "data/VOS2019")
	)
	if not data_root.is_absolute():
		data_root = PROJECT_ROOT / data_root

	train_split = str(data_cfg.get("train_split", "train"))
	valid_split = str(data_cfg.get("valid_split", "valid"))

	image_size_cfg = data_cfg.get("image_size", [384, 640])
	height = int(_resolve_setting(args.height, image_size_cfg[0], 384))
	width = int(_resolve_setting(args.width, image_size_cfg[1], 640))

	epochs = int(_resolve_setting(args.epochs, train_cfg.get("epochs"), 5))
	batch_size = int(_resolve_setting(args.batch_size, train_cfg.get("batch_size"), 2))
	lr = float(_resolve_setting(args.lr, train_cfg.get("lr"), 1e-4))
	weight_decay = float(
		_resolve_setting(args.weight_decay, train_cfg.get("weight_decay"), 1e-4)
	)
	num_workers = int(
		_resolve_setting(args.num_workers, train_cfg.get("num_workers"), 4)
	)
	print_freq = int(_resolve_setting(args.print_freq, train_cfg.get("print_freq"), 100))
	aux_weight = float(_resolve_setting(args.aux_weight, model_cfg.get("aux_weight"), 0.4))
	max_train_samples = _resolve_setting(
		args.max_train_samples, data_cfg.get("max_train_samples"), None
	)
	max_valid_samples = _resolve_setting(
		args.max_valid_samples, data_cfg.get("max_valid_samples"), None
	)

	device = _resolve_device(_resolve_setting(args.device, train_cfg.get("device"), "auto"))

	save_dir = Path(
		_resolve_setting(
			args.save_dir,
			output_cfg.get("save_dir"),
			f"outputs/{cfg.get('experiment_name', 'deeplabv3_vos2019_resnet50')}",
		)
	)
	if not save_dir.is_absolute():
		save_dir = PROJECT_ROOT / save_dir
	save_dir.mkdir(parents=True, exist_ok=True)

	train_dir = data_root / train_split
	valid_dir = data_root / valid_split

	train_ds = VOS2019BinarySegmentationDataset(
		split_dir=train_dir,
		image_size=(height, width),
		max_samples=max_train_samples,
		seed=seed,
	)
	valid_ds = VOS2019BinarySegmentationDataset(
		split_dir=valid_dir,
		image_size=(height, width),
		max_samples=max_valid_samples,
		seed=seed,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=(device.type == "cuda"),
	)
	valid_loader = DataLoader(
		valid_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=(device.type == "cuda"),
	)

	num_classes = int(model_cfg.get("num_classes", 2))
	pretrained_backbone = bool(model_cfg.get("pretrained_backbone", True))
	model = make_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=lr, weight_decay=weight_decay
	)
	backbone_name = str(model_cfg.get("backbone", "resnet50"))

	print(
		f"Dataset size: train={len(train_ds)} valid={len(valid_ds)} | "
		f"device={device}"
	)
	print(f"Backbone: {backbone_name} | Model: DeepLabV3")
	print(f"Using config: {args.config}")

	best_iou = -1.0
	history: List[Dict[str, float]] = []
	resolved_config: Dict[str, Any] = {
		"seed": seed,
		"data": {
			"data_root": str(data_root),
			"train_split": train_split,
			"valid_split": valid_split,
			"image_size": [height, width],
			"max_train_samples": max_train_samples,
			"max_valid_samples": max_valid_samples,
		},
		"model": {
			"arch": model_cfg.get("arch", "deeplabv3_resnet50"),
			"backbone": backbone_name,
			"num_classes": num_classes,
			"pretrained_backbone": pretrained_backbone,
			"aux_weight": aux_weight,
		},
		"training": {
			"epochs": epochs,
			"batch_size": batch_size,
			"lr": lr,
			"weight_decay": weight_decay,
			"num_workers": num_workers,
			"print_freq": print_freq,
			"device": str(device),
		},
		"output": {"save_dir": str(save_dir)},
	}

	for epoch in range(1, epochs + 1):
		print(f"\nEpoch {epoch}/{epochs}")
		train_stats = train_one_epoch(
			model=model,
			loader=train_loader,
			optimizer=optimizer,
			device=device,
			aux_weight=aux_weight,
			print_freq=max(1, print_freq),
		)
		val_stats = evaluate(model=model, loader=valid_loader, device=device)

		row = {
			"epoch": epoch,
			"train_loss": train_stats["loss"],
			"val_loss": val_stats["loss"],
			"val_pixel_acc": val_stats["pixel_acc"],
			"val_iou_fg": val_stats["iou_fg"],
		}
		history.append(row)

		print(
			f"[epoch {epoch}] train_loss={row['train_loss']:.4f} "
			f"val_loss={row['val_loss']:.4f} "
			f"val_pixel_acc={row['val_pixel_acc']:.4f} "
			f"val_iou_fg={row['val_iou_fg']:.4f}"
		)

		ckpt_last = save_dir / "deeplabv3_resnet50_last.pt"
		torch.save(
			{
				"epoch": epoch,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"config": resolved_config,
				"metrics": row,
			},
			ckpt_last,
		)

		if row["val_iou_fg"] > best_iou:
			best_iou = row["val_iou_fg"]
			ckpt_best = save_dir / "deeplabv3_resnet50_best.pt"
			torch.save(
				{
					"epoch": epoch,
					"model_state": model.state_dict(),
					"optimizer_state": optimizer.state_dict(),
					"config": resolved_config,
					"metrics": row,
				},
				ckpt_best,
			)
			print(f"Saved new best model @ {ckpt_best} (val_iou_fg={best_iou:.4f})")

	metrics_path = save_dir / "metrics.json"
	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(history, f, indent=2)
	print(f"Training done. Metrics saved to {metrics_path}")


if __name__ == "__main__":
	main()
