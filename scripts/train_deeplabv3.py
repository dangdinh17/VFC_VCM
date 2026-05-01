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
from unittest import loader

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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


def _compute_auto_class_weights(
	dataset: VSPWDataset,
	num_classes: int,
	ignore_index: int = 255,
	power: float = 0.5,
	min_weight: float = 0.1,
	max_weight: float = 10.0,
) -> torch.Tensor:
	counts = torch.zeros(num_classes, dtype=torch.float64)
	for item in tqdm(dataset.items, desc="Counting class pixels"):
		with Image.open(item.mask_path) as mask_img:
			mask = torch.as_tensor(np.array(mask_img), dtype=torch.long)
		valid = (mask >= 0) & (mask < num_classes) & (mask != ignore_index)
		if valid.any():
			counts += torch.bincount(mask[valid].reshape(-1), minlength=num_classes).double()

	present = counts > 0
	if not present.any():
		raise RuntimeError("Cannot compute class weights: no valid class pixels found.")

	freq = counts[present] / counts[present].sum().clamp_min(1.0)
	median_freq = freq.median()
	weights = torch.zeros(num_classes, dtype=torch.float32)
	weights[present] = (median_freq / freq).pow(power).float()
	weights[present] = weights[present].clamp(min=min_weight, max=max_weight)
	weights[present] /= weights[present].mean().clamp_min(1e-12)
	return weights


def _resolve_class_weights(
	class_weights_cfg: Any,
	dataset: VSPWDataset,
	num_classes: int,
	device: torch.device,
	ignore_index: int = 255,
	power: float = 0.5,
	min_weight: float = 0.1,
	max_weight: float = 10.0,
	cache_path: Path | None = None,
) -> torch.Tensor | None:
	if class_weights_cfg in (None, False, "none", "null"):
		return None
	if isinstance(class_weights_cfg, str) and class_weights_cfg.lower() == "auto":
		if cache_path is not None and cache_path.exists():
			weights = torch.load(cache_path, map_location="cpu")
			weights = torch.as_tensor(weights, dtype=torch.float32)
			if weights.numel() != num_classes:
				raise ValueError(
					f"Cached class weights at {cache_path} have {weights.numel()} values, "
					f"expected {num_classes}."
				)
			print(f"Loaded class weights from {cache_path}")
		else:
			weights = _compute_auto_class_weights(
				dataset=dataset,
				num_classes=num_classes,
				ignore_index=ignore_index,
				power=power,
				min_weight=min_weight,
				max_weight=max_weight,
			)
			if cache_path is not None:
				cache_path.parent.mkdir(parents=True, exist_ok=True)
				torch.save(weights.cpu(), cache_path)
				print(f"Saved class weights to {cache_path}")
	elif isinstance(class_weights_cfg, (list, tuple)):
		if len(class_weights_cfg) != num_classes:
			raise ValueError(
				f"training.class_weights must have {num_classes} values, "
				f"got {len(class_weights_cfg)}"
			)
		weights = torch.tensor(class_weights_cfg, dtype=torch.float32)
	else:
		raise ValueError(
			"training.class_weights must be null, 'auto', or a list of per-class weights"
		)

	print(
		"Class weights enabled | "
		f"min={weights[weights > 0].min().item():.4f} "
		f"max={weights.max().item():.4f} "
		f"mean_present={weights[weights > 0].mean().item():.4f}"
	)
	return weights.to(device)

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, ignore_index: int = 255):
	model.eval()
	total_loss = 0.0
	evaluator = EvaluatorTorch(num_classes)
	evaluator.reset()
	pbar = tqdm(loader)
	for batch in pbar:
		images = batch['image'].to(device)
		targets = batch['mask'].to(device)
		logits = model(images)["out"]
		loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
		total_loss += loss.item() * images.size(0)
		preds = logits.argmax(dim=1)
		evaluator.add_batch(targets.cpu(), preds.cpu(), ignore_index=ignore_index)

	evaluator.beforeval()
	gt_pixels_per_class = evaluator.confusion_matrix.sum(dim=1)
	# print("Classes xuất hiện:", (gt_pixels_per_class > 0).sum())
	# for i, v in enumerate(gt_pixels_per_class):
	# 	if v > 0:
	# 		print(i, v.item())
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
	class_weights: torch.Tensor | None = None,
	scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Dict[str, float]:
	model.train()
	total_loss = 0.0

	pbar = tqdm(loader)

	for step, batch in enumerate(pbar, start=1):
		images = batch["image"].to(device, non_blocking=True)
		targets = batch["mask"].to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)

		# out = model(images)
		# print("num_classes:", out["out"].shape[1])
		# print("target max:", targets.max())
		# print("target min:", targets.min())
		# print("out max:", out["out"].max())
		# print("out min:", out["out"].min())
		with torch.cuda.amp.autocast():
			out = model(images)

			# print(out['out'].shape)
			main_loss = F.cross_entropy(
				out["out"], targets, weight=class_weights, ignore_index=255
			)
			aux_loss = (
				F.cross_entropy(out["aux"], targets, weight=class_weights, ignore_index=255)
				if "aux" in out
				else 0.0
			)

			loss = main_loss + aux_weight * aux_loss if isinstance(aux_loss, torch.Tensor) else main_loss

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		if scheduler is not None:
			scheduler.step()



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
	parser.add_argument("--backbone-lr", type=float, default=None)
	parser.add_argument("--classifier-lr", type=float, default=None)
	parser.add_argument(
		"--lr-scheduler",
		type=str,
		default=None,
		choices=("none", "cosine"),
		help="Learning rate scheduler to use",
	)
	parser.add_argument(
		"--min-lr",
		type=float,
		default=None,
		help="Minimum learning rate for cosine annealing",
	)
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
	classifier_lr = float(
		_resolve_setting(
			args.classifier_lr,
			_resolve_setting(args.lr, train_cfg.get("classifier_lr"), train_cfg.get("lr")),
			1e-3,
		)
	)
	backbone_lr = float(
		_resolve_setting(args.backbone_lr, train_cfg.get("backbone_lr"), 1e-5)
	)
	lr_scheduler_name = str(
		_resolve_setting(args.lr_scheduler, train_cfg.get("lr_scheduler"), "cosine")
	).lower()
	min_lr = float(_resolve_setting(args.min_lr, train_cfg.get("min_lr"), 1e-6))
	weight_decay = float(
		_resolve_setting(args.weight_decay, train_cfg.get("weight_decay"), 1e-4)
	)
	num_workers = int(
		_resolve_setting(args.num_workers, train_cfg.get("num_workers"), 4)
	)
	print_freq = int(_resolve_setting(args.print_freq, train_cfg.get("print_freq"), 100))
	aux_weight = float(_resolve_setting(args.aux_weight, model_cfg.get("aux_weight"), 0.4))
	num_classes = int(model_cfg.get("num_classes", 2))
	class_weights_cfg = train_cfg.get("class_weights", None)
	class_weight_power = float(train_cfg.get("class_weight_power", 0.5))
	class_weight_min = float(train_cfg.get("class_weight_min", 0.1))
	class_weight_max = float(train_cfg.get("class_weight_max", 10.0))
	class_weights_cache_cfg = train_cfg.get("class_weights_cache", "class_weights.pt")
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
	class_weights_cache = None
	if class_weights_cache_cfg:
		class_weights_cache = Path(class_weights_cache_cfg)
		if not class_weights_cache.is_absolute():
			class_weights_cache = save_dir / class_weights_cache

	train_ds = VSPWDataset(
		root=data_root,
		split_dir=train_split,
		num_classes=num_classes,
		image_size=(height, width),
		max_samples=max_train_samples,
		seed=seed,
		seq_len=data_cfg.get("seq_len_train", 8)
	)
	valid_ds = VSPWDataset(
		root=data_root,
		split_dir=valid_split,
		num_classes=num_classes,
		image_size=(height, width),
		max_samples=max_valid_samples,
		seed=seed,
		seq_len=data_cfg.get("seq_len_valid", 3)
	)
	# all_labels = set()

	# for i in tqdm(range(len(train_ds))):
	# 	mask = train_ds[i]["mask"]
	# 	uniques = torch.unique(mask)
	# 	all_labels.update(uniques.tolist())

	# all_labels = sorted(all_labels)

	# print("Total unique labels:", len(all_labels))
	# print(all_labels)

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
	class_weights = _resolve_class_weights(
		class_weights_cfg=class_weights_cfg,
		dataset=train_ds,
		num_classes=num_classes,
		device=device,
		ignore_index=255,
		power=class_weight_power,
		min_weight=class_weight_min,
		max_weight=class_weight_max,
		cache_path=class_weights_cache,
	)

	pretrained_backbone = bool(model_cfg.get("pretrained_backbone", True))
	print(num_classes, pretrained_backbone)

	model = make_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone).to(device)

	# for name, p in model.named_parameters():
	# 	print(name, p.requires_grad)

	head_params = list(model.classifier.parameters())
	if model.aux_classifier is not None:
		head_params += list(model.aux_classifier.parameters())
	optimizer = torch.optim.AdamW(
		[
			{"params": model.backbone.parameters(), "lr": backbone_lr, "name": "backbone"},
			{"params": head_params, "lr": classifier_lr, "name": "classifier"},
		],
		weight_decay=weight_decay,
	)
	scheduler_t_max = epochs * len(train_loader)
	if lr_scheduler_name == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=scheduler_t_max,
			eta_min=min_lr,
		)
	elif lr_scheduler_name == "none":
		scheduler = None
	else:
		raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler_name}")
	backbone_name = str(model_cfg.get("backbone", "resnet50"))

	print(
		f"Dataset size: train={len(train_ds)} valid={len(valid_ds)} | "
		f"device={device}"
	)
	print(f"Backbone: {backbone_name} | Model: DeepLabV3")
	print(f"Using config: {args.config}")
	print(
		f"LR scheduler: {lr_scheduler_name} "
		f"(backbone_lr={backbone_lr:.2e}, classifier_lr={classifier_lr:.2e}, "
		f"min_lr={min_lr:.2e}, T_max_steps={scheduler_t_max})"
	)

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
			"backbone_lr": backbone_lr,
			"classifier_lr": classifier_lr,
			"lr_scheduler": lr_scheduler_name,
			"lr_scheduler_step": "batch",
			"lr_scheduler_t_max_steps": scheduler_t_max,
			"min_lr": min_lr,
			"weight_decay": weight_decay,
			"num_workers": num_workers,
			"print_freq": print_freq,
			"class_weights": class_weights_cfg,
			"class_weight_power": class_weight_power,
			"class_weight_min": class_weight_min,
			"class_weight_max": class_weight_max,
			"class_weights_cache": str(class_weights_cache) if class_weights_cache else None,
			"device": str(device),
		},
		"output": {"save_dir": str(save_dir)},
	}

	for epoch in range(1, epochs + 1):
		current_backbone_lr = optimizer.param_groups[0]["lr"]
		current_classifier_lr = optimizer.param_groups[1]["lr"]
		print(f"\nEpoch {epoch}/{epochs}")
		model.train()
		train_stats = train_one_epoch(
			model=model,
			loader=train_loader,
			optimizer=optimizer,
			device=device,
			aux_weight=aux_weight,
			print_freq=max(1, print_freq),
			class_weights=class_weights,
			scheduler=scheduler,
		)
		val_stats = evaluate(model=model, loader=valid_loader, device=device, num_classes=num_classes)

		next_backbone_lr = optimizer.param_groups[0]["lr"]
		next_classifier_lr = optimizer.param_groups[1]["lr"]

		row = {
			"epoch": epoch,
			"train_loss": train_stats["loss"],
			"val_loss": val_stats["loss"],
			"val_pixel_acc": val_stats["pixel_acc"],
			"val_miou": val_stats["miou"],
			"val_iou_valid_classes": val_stats.get("num_valid_iou_classes", 0),
			"backbone_lr": current_backbone_lr,
			"classifier_lr": current_classifier_lr,
			"next_backbone_lr": next_backbone_lr,
			"next_classifier_lr": next_classifier_lr,
		}
		history.append(row)

		print(
			f"[epoch {epoch}] train_loss={row['train_loss']:.4f} "
			f"val_loss={row['val_loss']:.4f} "
			f"val_pixel_acc={row['val_pixel_acc']:.4f} "
			f"val_miou={row['val_miou']:.4f} "
			f"val_iou_valid_classes={row['val_iou_valid_classes']} "
			f"backbone_lr={row['backbone_lr']:.2e}->{row['next_backbone_lr']:.2e} "
			f"classifier_lr={row['classifier_lr']:.2e}->{row['next_classifier_lr']:.2e}"
		)

		log_path = save_dir / "train_log.txt"
		with log_path.open("a", encoding="utf-8") as f:
			f.write(
				f"epoch={epoch} "
				f"train_loss={row['train_loss']:.6f} "
				f"val_loss={row['val_loss']:.6f} "
				f"val_pixel_acc={row['val_pixel_acc']:.6f} "
				f"val_miou={row['val_miou']:.6f} "
				f"backbone_lr={row['backbone_lr']:.8e} "
				f"classifier_lr={row['classifier_lr']:.8e} "
				f"next_backbone_lr={row['next_backbone_lr']:.8e} "
				f"next_classifier_lr={row['next_classifier_lr']:.8e}\n"
			)

		ckpt_last = save_dir / "deeplabv3_resnet50_last.pt"
		torch.save(
			{
				"epoch": epoch,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"scheduler_state": scheduler.state_dict() if scheduler is not None else None,
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
					"scheduler_state": scheduler.state_dict() if scheduler is not None else None,
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
