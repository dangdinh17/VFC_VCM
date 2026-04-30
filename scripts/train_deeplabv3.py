from __future__ import annotations

import argparse
import json
import os
from pyexpat import model
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import device, nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.datasets import VSPWFrameLMDBDataset, VSPWDataset
from src.evaluate import EvaluatorTorch

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
		weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
		weights_backbone=backbone_weights,
		aux_loss=True,
	)
	# model.classifier = DeepLabHead(256, num_classes)
	# model.aux_classifier = DeepLabHead(256, num_classes)
	
    
    
	# thay head cho số lớp mới
	model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
	if model.aux_classifier is not None:
		model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
	

	# Freeze backbone feature extractor (ResNet-50) for this training run.
	# for p in model.backbone.parameters():
	# 	p.requires_grad = False
	return model

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, ignore_index: int = 255):
	model.eval()
	total_loss = 0.0
	evaluator = EvaluatorTorch(num_classes)
	evaluator.reset()

	for batch in loader:
		images = batch['image'].to(device)
		targets = batch['mask'].to(device)
		logits = model(images)["out"]
		loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
		total_loss += loss.item() * images.size(0)
		preds = logits.argmax(dim=1)
		evaluator.add_batch(targets.cpu(), preds.cpu(), ignore_index=ignore_index)

	evaluator.beforeval()
	inter = torch.diag(evaluator.confusion_matrix)
	union = evaluator.confusion_matrix.sum(dim=1) + evaluator.confusion_matrix.sum(dim=0) - inter
	per_class_iou = (inter / torch.clamp(union, min=1.0)).cpu().numpy()
	valid_classes = (evaluator.confusion_matrix.sum(dim=1) > 0)

	avg_loss = total_loss / len(loader.dataset)

	return {
		"loss": avg_loss,
		"pixel_acc": evaluator.pixel_accuracy(),
		"pixel_acc_class": evaluator.pixel_accuracy_class(),
		"miou": evaluator.mean_iou(),
		"fw_iou": evaluator.fw_iou(),
		"per_class_iou": per_class_iou,
		"num_valid_iou_classes": int(valid_classes.sum().item()),
	}

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

	pbar = tqdm(loader)

	for step, batch in enumerate(pbar, start=1):
		images = batch["image"].to(device, non_blocking=True)
		targets = batch["mask"].to(device, non_blocking=True)
  
		# out = model(images)
		# print("num_classes:", out["out"].shape[1])
		# print("target max:", targets.max())
		# print("target min:", targets.min())
		# print("out max:", out["out"].max())
		# print("out min:", out["out"].min())
		with torch.cuda.amp.autocast():
			out = model(images)
			
			# print(out['out'].shape)
			main_loss = F.cross_entropy(out["out"], targets, ignore_index=255)
			aux_loss = F.cross_entropy(out["aux"], targets, ignore_index=255) if "aux" in out else 0.0

			loss = main_loss + aux_weight * aux_loss if isinstance(aux_loss, torch.Tensor) else main_loss

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

  
		
		total_loss += float(loss.item()) * images.size(0)

		# ✅ thêm dòng này
		pbar.set_postfix(
			loss=f"{loss.item():.4f}",
			main=f"{main_loss.item():.4f}",
		)
		
		# if step % print_freq == 0 or step == len(loader):
		# 	print(
		# 		f"[train] step {step:04d}/{len(loader):04d} "
		# 		f"loss={loss.item():.4f} main={main_loss.item():.4f}"
		# 	)

	avg_loss = total_loss / max(1, len(loader.dataset))
	return {"loss": avg_loss}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser("Train DeepLabV3-ResNet50 on VOS2019")
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("configs/vspw/deeplabv3_resnet50.yaml"),
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
	torch.backends.cudnn.benchmark = True
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True
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
	num_classes = int(model_cfg.get("num_classes", 2))
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

	# VSPW layout uses data_root/data/<video> with split lists at data_root/*.txt
	if (data_root / "data").exists():
		train_dir = data_root
		valid_dir = data_root
		vspw_train = train_split
		vspw_valid = valid_split
	else:
		train_dir = data_root / train_split
		valid_dir = data_root / valid_split
		vspw_train = None
		vspw_valid = None

	train_ds = VSPWDataset(
		split_dir=train_dir,
		num_classes=num_classes,
		image_size=(height, width),
		max_samples=max_train_samples,
		seed=seed,
	)
	valid_ds = VSPWDataset(
		split_dir=valid_dir,
		num_classes=num_classes,
		image_size=(height, width),
		max_samples=max_valid_samples,
		seed=seed,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		persistent_workers=True,
		prefetch_factor=4,
		num_workers=num_workers,
		pin_memory=(device.type == "cuda"),
	)
	valid_loader = DataLoader(
		valid_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		persistent_workers=True,
		prefetch_factor=4,
		pin_memory=(device.type == "cuda"),
	)

	pretrained_backbone = bool(model_cfg.get("pretrained_backbone", True))
	print(num_classes, pretrained_backbone)
	
	model = make_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone).to(device)
	
	for name, p in model.named_parameters():
		print(name, p.requires_grad)

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

	global scaler
	scaler = torch.cuda.amp.GradScaler()

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
		model.train()
		train_stats = train_one_epoch(
			model=model,
			loader=train_loader,
			optimizer=optimizer,
			device=device,
			aux_weight=aux_weight,
			print_freq=max(1, print_freq),
		)
		val_stats = evaluate(model=model, loader=valid_loader, device=device, num_classes=num_classes)

		row = {
			"epoch": epoch,
			"train_loss": train_stats["loss"],
			"val_loss": val_stats["loss"],
			"val_pixel_acc": val_stats["pixel_acc"],
			"val_miou": val_stats["miou"],
			"val_iou_valid_classes": val_stats.get("num_valid_iou_classes", 0),
		}
		history.append(row)

		print(
			f"[epoch {epoch}] train_loss={row['train_loss']:.4f} "
			f"val_loss={row['val_loss']:.4f} "
			f"val_pixel_acc={row['val_pixel_acc']:.4f} "
			f"val_miou={row['val_miou']:.4f} "
			f"val_iou_valid_classes={row['val_iou_valid_classes']}"
		)

		log_path = save_dir / "train_log.txt"
		with log_path.open("a", encoding="utf-8") as f:
			f.write(
				f"epoch={epoch} "
				f"train_loss={row['train_loss']:.6f} "
				f"val_loss={row['val_loss']:.6f} "
				f"val_pixel_acc={row['val_pixel_acc']:.6f} "
				f"val_miou={row['val_miou']:.6f}\n"
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

		if row["val_miou"] > best_iou:
			best_iou = row["val_miou"]
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
			print(f"Saved new best model @ {ckpt_best} (val_miou={best_iou:.4f})")

	metrics_path = save_dir / "metrics.json"
	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(history, f, indent=2)
	print(f"Training done. Metrics saved to {metrics_path}")
	print(f"Training log saved to {save_dir / 'train_log.txt'}")


if __name__ == "__main__":
	main()
