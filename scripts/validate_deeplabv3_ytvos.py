from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import VSPWDataset
from src.models.backbone import FeatureExtraction
from src.utils import AverageMeter, load_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_setting(cli_val: Any, cfg_val: Any, default_val: Any) -> Any:
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return cfg_val
    return default_val


def _resolve_device(name: Optional[str]) -> torch.device:
    if name in (None, "auto"):
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _compute_binary_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    pred_fg = pred == 1
    gt_fg = target == 1
    inter = (pred_fg & gt_fg).sum().item()
    union = (pred_fg | gt_fg).sum().item()
    return float(inter) / float(max(union, 1))


def _compute_fg_ratio(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float((mask == 1).float().mean().item())


class DeepLabFeatureAdapter(nn.Module):
    """Run DeepLabV3 from res2 feature input (C=256)."""

    def __init__(self, deeplab_model: nn.Module) -> None:
        super().__init__()
        self.layer2 = deeplab_model.backbone.layer2
        self.layer3 = deeplab_model.backbone.layer3
        self.layer4 = deeplab_model.backbone.layer4
        self.classifier = deeplab_model.classifier
        self.aux_classifier = deeplab_model.aux_classifier

    def forward(self, res2_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        x2 = self.layer2(res2_feature)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        outputs = {"out": self.classifier(x4)}
        if self.aux_classifier is not None:
            outputs["aux"] = self.aux_classifier(x3)
        return outputs


def _build_deeplab_feature_adapter(
    *,
    pretrained: bool,
    device: torch.device,
    num_classes: int,
    weights_path: Optional[Path],
) -> DeepLabFeatureAdapter:
    deeplab = deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
        num_classes=num_classes,
        aux_loss=True,
    ).to(device)

    if weights_path is not None:
        ckpt_path = weights_path if weights_path.is_absolute() else PROJECT_ROOT / weights_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DeepLab checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        incompatible = deeplab.load_state_dict(state_dict, strict=False)
        print(f"Loaded DeepLab weights from: {ckpt_path}")
        if len(incompatible.missing_keys) > 0:
            print(f"[warn] DeepLab missing keys: {len(incompatible.missing_keys)}")
        if len(incompatible.unexpected_keys) > 0:
            print(f"[warn] DeepLab unexpected keys: {len(incompatible.unexpected_keys)}")
    else:
        print("[warn] No DeepLab weights provided; using random initialization.")

    adapter = DeepLabFeatureAdapter(deeplab).to(device)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False
    return adapter


@torch.no_grad()
def validate_deeplab_baseline(
    *,
    dataloader: DataLoader,
    feature_extractor: FeatureExtraction,
    deeplab_adapter: DeepLabFeatureAdapter,
    device: torch.device,
    aux_weight: float,
    use_tqdm: bool,
) -> Dict[str, float]:
    feature_extractor.eval()
    deeplab_adapter.eval()

    totals = {
        "l_task": AverageMeter(),
        "iou_fg": AverageMeter(),
        "fg_ratio_gt": AverageMeter(),
        "fg_ratio_pred": AverageMeter(),
    }

    iterator = tqdm(dataloader, desc="deeplabv3_val", leave=False) if use_tqdm else dataloader
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        has_time = image.dim() == 5
        steps = image.shape[1] if has_time else 1

        for t in range(steps):
            img_t = image[:, t] if has_time else image
            mask_t = mask[:, t] if has_time else mask

            src_feat = feature_extractor(img_t)
            seg_out = deeplab_adapter(src_feat)
            logits = F.interpolate(seg_out["out"], size=mask_t.shape[-2:], mode="bilinear", align_corners=False)

            seg_loss = F.cross_entropy(logits, mask_t)
            if "aux" in seg_out:
                aux_logits = F.interpolate(seg_out["aux"], size=mask_t.shape[-2:], mode="bilinear", align_corners=False)
                seg_loss = seg_loss + aux_weight * F.cross_entropy(aux_logits, mask_t)

            pred = logits.argmax(dim=1)
            totals["l_task"].update(float(seg_loss.item()))
            totals["iou_fg"].update(float(_compute_binary_iou(logits, mask_t)))
            totals["fg_ratio_gt"].update(float(_compute_fg_ratio(mask_t)))
            totals["fg_ratio_pred"].update(float(_compute_fg_ratio(pred)))

    return {k: m.avg for k, m in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Validate DeepLabV3 baseline on YTVOS")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ytvos/roi_vfc_deeplabv3_ytvos_codec_bitrate16.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--use-tqdm", action="store_true")
    parser.add_argument("--deeplab-weights", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Force resize (no random/center crop) for full-image validation",
    )
    return parser.parse_args()


def _read_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_config(str(config_path)) or {}


def main() -> None:
    args = parse_args()
    cfg = _read_config(args.config)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    seed = int(_resolve_setting(args.seed, cfg.get("seed"), 42))
    seed_everything(seed)

    device = _resolve_device(_resolve_setting(args.device, train_cfg.get("device"), "auto"))

    data_root = Path(_resolve_setting(args.data_root, data_cfg.get("data_root"), "data/VOS2019"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    image_size = data_cfg.get("image_size", [384, 640])
    spatial_transform = str(data_cfg.get("spatial_transform", "crop")).lower()
    valid_crop_type = str(data_cfg.get("valid_crop_type", "center")).lower()
    pad_if_needed = bool(data_cfg.get("pad_if_needed", True))
    seq_len = int(data_cfg.get("seq_len", 1))
    seq_stride = int(data_cfg.get("seq_stride", data_cfg.get("seq_len", 1)))
    max_valid_samples = _resolve_setting(args.max_valid_samples, data_cfg.get("max_valid_samples"), None)

    valid_dir = data_root / str(data_cfg.get("valid_split", "valid"))

    if args.no_crop:
        spatial_transform = "resize"
        valid_crop_type = "center"

    valid_ds = VSPWDataset(
        split_dir=valid_dir,
        image_size=(int(image_size[0]), int(image_size[1])),
        spatial_transform=spatial_transform,
        crop_type=valid_crop_type,
        pad_if_needed=pad_if_needed,
        max_samples=max_valid_samples,
        seed=seed,
        seq_len=seq_len,
        seq_stride=seq_stride,
    )

    batch_size = int(_resolve_setting(args.batch_size, train_cfg.get("valid_batch_size"), 1))
    num_workers = int(_resolve_setting(args.num_workers, train_cfg.get("num_workers"), 4))
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feat_cfg = model_cfg.get("feature_extraction", {})
    feature_extractor = FeatureExtraction(
        backbone_name=str(feat_cfg.get("backbone", "resnet50")),
        pretrained=bool(feat_cfg.get("pretrained", True)),
    ).to(device)

    deeplab_cfg = model_cfg.get("deeplabv3", {})
    weights_path = _resolve_setting(args.deeplab_weights, deeplab_cfg.get("weights_path"), None)
    deeplab_adapter = _build_deeplab_feature_adapter(
        pretrained=bool(deeplab_cfg.get("pretrained", True)),
        device=device,
        num_classes=int(deeplab_cfg.get("num_classes", 2)),
        weights_path=Path(weights_path) if weights_path else None,
    )

    aux_weight = float(deeplab_cfg.get("aux_weight", 0.4))
    use_tqdm = bool(args.use_tqdm)

    print(
        "valid={count} device={device} batch_size={bs} "
        "spatial_transform={spatial} crop={crop} max_valid_samples={mvs}"
        .format(
            count=len(valid_ds),
            device=device,
            bs=batch_size,
            spatial=spatial_transform,
            crop=valid_crop_type,
            mvs=max_valid_samples,
        )
    )
    metrics = validate_deeplab_baseline(
        dataloader=valid_loader,
        feature_extractor=feature_extractor,
        deeplab_adapter=deeplab_adapter,
        device=device,
        aux_weight=aux_weight,
        use_tqdm=use_tqdm,
    )

    print(
        "DeepLab baseline (src features) on val: "
        f"iou_fg={metrics['iou_fg']:.4f} "
        f"l_task={metrics['l_task']:.4f} "
        f"fg_gt={metrics['fg_ratio_gt']:.4f} "
        f"fg_pred={metrics['fg_ratio_pred']:.4f}"
    )

    if args.output_json:
        output_path = args.output_json
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(args.config),
            "seed": seed,
            "device": str(device),
            "weights_path": str(weights_path) if weights_path else None,
            "metrics": metrics,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
