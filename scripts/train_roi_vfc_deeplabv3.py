from __future__ import annotations
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import comet_ml
from comet_ml import ExistingExperiment, Experiment
import argparse
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import VSPWDataset
from src.models.backbone import FeatureExtraction, PerceptionExtraction
from src.models.feature_space_transfer import FeatureSpaceTransfer
from src.models.roi_vfc import ROI_VFC
from src.utils import load_config, AverageMeter


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


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


class DeepLabFeatureAdapter(nn.Module):
    """Run pretrained/frozen DeepLabV3 from res2 feature input (C=256)."""

    def __init__(self, deeplab_model: nn.Module) -> None:
        super().__init__()
        self.layer2 = deeplab_model.backbone.layer2
        self.layer3 = deeplab_model.backbone.layer3
        self.layer4 = deeplab_model.backbone.layer4
        self.classifier = deeplab_model.classifier
        self.aux_classifier = deeplab_model.aux_classifier

    def forward(self, res2_feature: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        x2 = self.layer2(res2_feature)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = self.classifier(x4)
        outputs: Dict[str, torch.Tensor] = {"out": out}
        if self.aux_classifier is not None:
            outputs["aux"] = self.aux_classifier(x3)
        if return_features:
            # High-level downstream backbone features for D_high
            outputs["high_features"] = [x3, x4]
        return outputs


def compute_codec_loss(
    roi_out: Dict[str, Any],
    weights: Dict[str, float],
    device: torch.device,
    *,
    rate_scale: float = 1.0,
    include_rz: bool = True,
) -> Dict[str, torch.Tensor]:
    """Paper Eq.(5): L_codec = λ_R(R_r+R_m) + λ_f D_f + λ_c D_c + λ_p D_p."""
    r_r = roi_out.get("bpp_y", torch.tensor(0.0, device=device))
    r_m = roi_out.get("bpp_z", torch.tensor(0.0, device=device))
    d_f = roi_out.get("mse_f", torch.tensor(0.0, device=device))
    d_c = roi_out.get("mse_c", torch.tensor(0.0, device=device))
    d_p = roi_out.get("mse_p", torch.tensor(0.0, device=device))

    if not torch.is_tensor(r_r):
        r_r = torch.tensor(float(r_r), device=device)
    if not torch.is_tensor(r_m):
        r_m = torch.tensor(float(r_m), device=device)
    if not torch.is_tensor(d_f):
        d_f = torch.tensor(float(d_f), device=device)
    if not torch.is_tensor(d_c):
        d_c = torch.tensor(float(d_c), device=device)
    if not torch.is_tensor(d_p):
        d_p = torch.tensor(float(d_p), device=device)

    rate_term = r_r + (r_m if include_rz else torch.zeros_like(r_m))
    lambda_r_eff = float(weights["lambda_r"]) * float(rate_scale)

    l_codec = (
        lambda_r_eff * rate_term
        + float(weights["lambda_f"]) * d_f
        + float(weights["lambda_c"]) * d_c
        + float(weights["lambda_p"]) * d_p
    )
    return {
        "l_codec": l_codec,
        "rate_term": rate_term,
        "lambda_r_eff": torch.tensor(lambda_r_eff, device=device),
        "r_r": r_r,
        "r_m": r_m,
        "d_f": d_f,
        "d_c": d_c,
        "d_p": d_p,
    }


def compute_fst_loss(
    *,
    l_task: torch.Tensor,
    d_x: torch.Tensor,
    d_mid: torch.Tensor,
    d_high: torch.Tensor,
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Paper Eq.(7): L_FST = λ_task L_task + λ_x D_x + λ_mid D_mid + λ_high D_high."""
    l_fst = (
        float(weights["lambda_task"]) * l_task
        + float(weights["lambda_x"]) * d_x
        + float(weights["lambda_mid"]) * d_mid
        + float(weights["lambda_high"]) * d_high
    )
    return {
        "l_fst": l_fst,
        "l_task": l_task,
        "d_x": d_x,
        "d_mid": d_mid,
        "d_high": d_high,
    }


def _build_deeplab_feature_adapter(
    pretrained: bool,
    device: torch.device,
    num_classes: int = 2,
    weights_path: Optional[Path] = None,
) -> DeepLabFeatureAdapter:
    # Use backbone pretraining by default for the feature-input downstream model.
    deeplab = deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
        num_classes=num_classes,
        aux_loss=True,
    ).to(device)

    if weights_path is not None:
        ckpt_path = weights_path if weights_path.is_absolute() else PROJECT_ROOT / weights_path
        if not ckpt_path.exists():
            # pass
            raise FileNotFoundError(f"DeepLab checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        incompatible = deeplab.load_state_dict(state_dict, strict=False)
        print(f"Loaded DeepLab weights from: {ckpt_path}")
        if len(incompatible.missing_keys) > 0:
            print(f"[warn] DeepLab missing keys: {len(incompatible.missing_keys)}")
        if len(incompatible.unexpected_keys) > 0:
            print(f"[warn] DeepLab unexpected keys: {len(incompatible.unexpected_keys)}")

    adapter = DeepLabFeatureAdapter(deeplab).to(device)
    _freeze_module(adapter)
    return adapter


def _build_comet_experiment(logging_cfg: Dict[str, Any], run_name: str):
    comet_cfg = logging_cfg.get("comet", {}) if isinstance(logging_cfg, dict) else {}
    if not comet_cfg.get("enabled", False):
        return None

    experiment_key = comet_cfg.get("experiment_key")
    if experiment_key and ExistingExperiment is not None:
        exp = ExistingExperiment(
            api_key=comet_cfg.get("api_key"),
            project_name=comet_cfg.get("project_name", "roi-vfc-segmentation"),
            workspace=comet_cfg.get("workspace"),
            previous_experiment=str(experiment_key),
            auto_output_logging="native",
        )
    else:
        exp = Experiment(
            api_key=comet_cfg.get("api_key"),
            project_name=comet_cfg.get("project_name", "roi-vfc-segmentation"),
            workspace=comet_cfg.get("workspace"),
            auto_output_logging="native",
        )
    exp.set_name(run_name)
    return exp

def _append_metrics(save_path: Path, history: List[Dict[str, Any]]) -> None:
    existing: List[Dict[str, Any]] = []
    if save_path.exists():
        try:
            with save_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    if not isinstance(existing, list):
        existing = []

    if len(existing) < len(history):
        existing.extend(history[len(existing) :])
    else:
        existing = history

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

def _compute_miou(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """
    logits: (B, C, H, W)
    target: (B, H, W)
    """

    pred = logits.argmax(dim=1)

    pred = pred.view(-1)
    target = target.view(-1)

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    iou_per_class = []

    for cls in range(num_classes):
        pred_c = pred == cls
        gt_c = target == cls

        inter = (pred_c & gt_c).sum().item()
        union = (pred_c | gt_c).sum().item()

        if union == 0:
            continue

        iou_per_class.append(inter / union)

    if len(iou_per_class) == 0:
        return 0.0

    return float(sum(iou_per_class) / len(iou_per_class))

@torch.no_grad()
def _validate_deeplab_baseline(
    *,
    dataloader: DataLoader,
    feature_extractor: FeatureExtraction,
    deeplab_adapter: DeepLabFeatureAdapter,
    device: torch.device,
    aux_weight: float,
    use_tqdm: bool,
    num_classes: int,
) -> Dict[str, float]:
    feature_extractor.eval()
    deeplab_adapter.eval()

    totals = {
        "l_task": AverageMeter(),
        "iou_fg": AverageMeter(),
        "fg_ratio_gt": AverageMeter(),
        "fg_ratio_pred": AverageMeter(),
    }

    iterator = dataloader
    if use_tqdm:
        iterator = tqdm(dataloader, desc="deeplab_baseline", leave=False)

    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        has_time = image.dim() == 5
        if has_time:
            T = image.shape[1]
        else:
            T = 1

        for t in range(T):
            img_t = image[:, t] if has_time else image
            mask_t = mask[:, t] if has_time else mask

            src_feat = feature_extractor(img_t)
            seg_out = deeplab_adapter(src_feat, return_features=False)
            logits = F.interpolate(seg_out["out"], size=mask_t.shape[-2:], mode="bilinear", align_corners=False)

            seg_loss = F.cross_entropy(logits, mask_t)
            if "aux" in seg_out:
                aux_logits = F.interpolate(seg_out["aux"], size=mask_t.shape[-2:], mode="bilinear", align_corners=False)
                seg_loss = seg_loss + aux_weight * F.cross_entropy(aux_logits, mask_t)

            pred = logits.argmax(dim=1)
            totals["l_task"].update(float(seg_loss.item()))
            totals["iou_fg"].update(float(_compute_miou(logits, mask_t, num_classes=num_classes)))


    return {k: m.avg for k, m in totals.items()}


def _build_dataloaders(cfg: Dict[str, Any], num_workers: int, train_bs: int, val_bs: int, seed: int):
    data_cfg = cfg.get("data", {})
    data_root = Path(data_cfg.get("data_root", "data/VOS2019"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    image_size = data_cfg.get("image_size", [384, 640])
    spatial_transform = str(data_cfg.get("spatial_transform", "crop")).lower()
    train_crop_type = str(data_cfg.get("train_crop_type", "random")).lower()
    valid_crop_type = str(data_cfg.get("valid_crop_type", "center")).lower()
    pad_if_needed = bool(data_cfg.get("pad_if_needed", True))
    # For VSPW the repository uses data_root/data/<video> and split lists at data_root/*.txt
    # detect VSPW-style by presence of data_root/data
    if (data_root / "data").exists():
        train_dir = data_root
        valid_dir = data_root
    else:
        train_dir = data_root / str(data_cfg.get("train_split", "train"))
        valid_dir = data_root / str(data_cfg.get("valid_split", "valid"))

    # train_ds = SemanticSegmentationDataset(
    #     split_dir=train_dir,
    #     vspw_split=(str(data_cfg.get("train_split")) if "vspw" in str(data_root.name).lower() or "vspw" in str(data_root) else None),
    #     image_size=(int(image_size[0]), int(image_size[1])),
    #     spatial_transform=spatial_transform,
    #     crop_type=train_crop_type,
    #     pad_if_needed=pad_if_needed,
    #     seq_len=int(data_cfg.get("seq_len", 1)),
    #     seq_stride=int(data_cfg.get("seq_stride", data_cfg.get("seq_len", 1))),
    #     max_samples=data_cfg.get("max_train_samples"),
    #     seed=seed,
    #     num_classes=int(data_cfg.get("num_classes", 124)),
    # )
    # val_ds = SemanticSegmentationDataset(
    #     split_dir=valid_dir,
    #     vspw_split=(str(data_cfg.get("valid_split")) if "vspw" in str(data_root.name).lower() or "vspw" in str(data_root) else None),
    #     image_size=(int(image_size[0]), int(image_size[1])),
    #     spatial_transform=spatial_transform,
    #     crop_type=valid_crop_type,
    #     pad_if_needed=pad_if_needed,
    #     seq_len=int(data_cfg.get("seq_len", 1)),
    #     seq_stride=int(data_cfg.get("seq_stride", data_cfg.get("seq_len", 1))),
    #     max_samples=data_cfg.get("max_valid_samples"),
    #     seed=seed,
    #     num_classes=int(data_cfg.get("num_classes", 124)),
    # )
    train_ds = VSPWDataset(
        split_dir=train_dir,
        vspw_split=(str(data_cfg.get("train_split")) if "vspw" in str(data_root.name).lower() or "vspw" in str(data_root) else None),
        image_size=(int(image_size[0]), int(image_size[1])),
        spatial_transform=spatial_transform,
        crop_type=train_crop_type,
        pad_if_needed=pad_if_needed,
        seq_len=int(data_cfg.get("seq_len", 1)),
        seq_stride=int(data_cfg.get("seq_stride", data_cfg.get("seq_len", 1))),
        max_samples=data_cfg.get("max_train_samples"),
        seed=seed,
        num_classes=int(data_cfg.get("num_classes", 124)),
    )
    val_ds = VSPWDataset(
        split_dir=valid_dir,
        vspw_split=(str(data_cfg.get("valid_split")) if "vspw" in str(data_root.name).lower() or "vspw" in str(data_root) else None),
        image_size=(int(image_size[0]), int(image_size[1])),
        spatial_transform=spatial_transform,
        crop_type=valid_crop_type,
        pad_if_needed=pad_if_needed,
        seq_len=int(data_cfg.get("seq_len", 1)),
        seq_stride=int(data_cfg.get("seq_stride", data_cfg.get("seq_len", 1))),
        max_samples=data_cfg.get("max_valid_samples"),
        seed=seed,
        num_classes=int(data_cfg.get("num_classes", 124)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def _build_training_stages(training_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build normalized stage list.

    If `training.stages` is missing, fallback to a 2-stage split:
    - stage1_codec: first half epochs, optimize codec
    - stage2_fst: second half epochs, optimize fst
    """
    default_epochs = int(training_cfg.get("epochs", 20))
    default_lr = float(training_cfg.get("lr", 1e-4))
    default_wd = float(training_cfg.get("weight_decay", 1e-4))

    stage_cfgs = training_cfg.get("stages")
    if not isinstance(stage_cfgs, list) or len(stage_cfgs) == 0:
        stage1_epochs = max(1, default_epochs // 2)
        stage2_epochs = max(1, default_epochs - stage1_epochs)
        return [
            {
                "name": "stage1_codec",
                "optimize": "codec",
                "epochs": stage1_epochs,
                "lr": default_lr,
                "weight_decay": default_wd,
                "codec": {"rate_scale": 1.0, "include_rz": True},
            },
            {
                "name": "stage2_fst",
                "optimize": "fst",
                "epochs": stage2_epochs,
                "lr": 1e-5,
                "weight_decay": default_wd,
                "codec": {"rate_scale": 1.0, "include_rz": True},
            },
        ]

    normalized: List[Dict[str, Any]] = []
    for idx, raw in enumerate(stage_cfgs, start=1):
        if not isinstance(raw, dict):
            continue
        optimize = str(raw.get("optimize", "codec")).lower()
        if optimize not in {"codec", "fst", "both"}:
            optimize = "codec"
        epochs = max(1, int(raw.get("epochs", 1)))
        stage_codec = raw.get("codec", {}) if isinstance(raw.get("codec", {}), dict) else {}
        normalized.append(
            {
                "name": str(raw.get("name", f"stage{idx}_{optimize}")),
                "optimize": optimize,
                "epochs": epochs,
                "lr": float(raw.get("lr", default_lr)),
                "weight_decay": float(raw.get("weight_decay", default_wd)),
                "codec": {
                    "rate_scale": float(stage_codec.get("rate_scale", 1.0)),
                    "include_rz": bool(stage_codec.get("include_rz", True)),
                },
            }
        )

    if len(normalized) == 0:
        raise ValueError("No valid training stages configured.")
    return normalized


def _locate_stage_for_epoch(stages: List[Dict[str, Any]], global_epoch: int) -> Dict[str, int]:
    """Map a 1-based global epoch index to stage index and local epoch."""
    if global_epoch <= 0:
        return {"stage_index": 1, "local_epoch": 1}
    acc = 0
    for idx, stage in enumerate(stages, start=1):
        stage_epochs = max(1, int(stage.get("epochs", 1)))
        acc += stage_epochs
        if global_epoch <= acc:
            local_epoch = global_epoch - (acc - stage_epochs)
            return {"stage_index": idx, "local_epoch": local_epoch}
    # fallback: clamp to last stage
    last_stage_epochs = max(1, int(stages[-1].get("epochs", 1))) if stages else 1
    return {"stage_index": len(stages) if stages else 1, "local_epoch": last_stage_epochs}


def _set_module_trainability(
    module: nn.Module,
    original_requires_grad: Dict[str, bool],
    enabled: bool,
) -> None:
    for name, p in module.named_parameters():
        base = bool(original_requires_grad.get(name, p.requires_grad))
        p.requires_grad = base if enabled else False


def _build_optimizer(
    *,
    roi_vfc: ROI_VFC,
    feature_space_transfer: FeatureSpaceTransfer,
    stage_optimize: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params: List[torch.nn.Parameter] = []
    if stage_optimize in {"codec", "both"}:
        params.extend([p for p in roi_vfc.parameters() if p.requires_grad])
    if stage_optimize in {"fst", "both"}:
        params.extend([p for p in feature_space_transfer.parameters() if p.requires_grad])

    if len(params) == 0:
        raise RuntimeError(f"No trainable parameters found for stage optimize='{stage_optimize}'.")

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _run_epoch(
    *,
    mode: str,
    dataloader: DataLoader,
    roi_vfc: ROI_VFC,
    feature_extractor: FeatureExtraction,
    feature_space_transfer: FeatureSpaceTransfer,
    deeplab_adapter: DeepLabFeatureAdapter,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    aux_weight: float,
    codec_weights: Dict[str, float],
    fst_weights: Dict[str, float],
    use_tqdm: bool,
    comet_exp: Optional[Any] = None,
    epoch: int = 0,
    split: str = "train",
    log_batch_metrics: bool = False,
    batch_log_interval: int = 1,
    optimize_target: str = "both",
    codec_stage_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    is_train = mode == "train"
    train_roi_vfc = is_train and optimize_target in {"codec", "both"}
    train_fst = is_train and optimize_target in {"fst", "both"}

    roi_vfc.train(train_roi_vfc)
    feature_space_transfer.train(train_fst)
    feature_extractor.eval()
    deeplab_adapter.eval()

    totals = {
        "total_loss": AverageMeter(),
        "l_codec": AverageMeter(),
        "l_fst": AverageMeter(),
        "l_task": AverageMeter(),
        "d_x": AverageMeter(),
        "d_mid": AverageMeter(),
        "d_high": AverageMeter(),
        "r_r": AverageMeter(),
        "r_m": AverageMeter(),
        "d_f": AverageMeter(),
        "d_c": AverageMeter(),
        "d_p": AverageMeter(),
        "rate_term": AverageMeter(),
        "feature_loss": AverageMeter(),
        "iou_fg": AverageMeter(),
        "fg_ratio_gt": AverageMeter(),
        "fg_ratio_pred": AverageMeter(),
    }
    num_batches = 0

    iterator = dataloader
    if use_tqdm:
        iterator = tqdm(dataloader, desc=f"{mode}", leave=False)

    total_batches = max(len(dataloader), 1)
    codec_stage_cfg = codec_stage_cfg or {}
    stage_rate_scale = float(codec_stage_cfg.get("rate_scale", 1.0))
    stage_include_rz = bool(codec_stage_cfg.get("include_rz", True))

    for batch_idx, batch in enumerate(iterator, start=1):
        # Reset ROI feature buffer at the start of each batch/sequence
        if hasattr(roi_vfc, "reset_buffer") and callable(getattr(roi_vfc, "reset_buffer")):
            roi_vfc.reset_buffer()

        image = batch["image"].to(device, non_blocking=True)
        image_raw = batch["image_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        # Determine if input contains a temporal dimension: (B, T, C, H, W)
        has_time = image.dim() == 5
        if has_time:
            # ensure shapes
            B, T = image.shape[0], image.shape[1]
        else:
            B, T = image.shape[0], 1

        # precompute feature extractor outputs per frame (no grad)
        src_feats = []
        if has_time:
            with torch.no_grad():
                for t in range(T):
                    src_feats.append(feature_extractor(image[:, t]))
        else:
            with torch.no_grad():
                src_feats.append(feature_extractor(image))

        # Zero optimizer grads once per batch
        if is_train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        # Accumulate per-frame losses/metrics then average over time
        batch_meters = {
            "total_loss": AverageMeter(),
            "l_codec": AverageMeter(),
            "l_fst": AverageMeter(),
            "l_task": AverageMeter(),
            "d_x": AverageMeter(),
            "d_mid": AverageMeter(),
            "d_high": AverageMeter(),
            "r_r": AverageMeter(),
            "r_m": AverageMeter(),
            "d_f": AverageMeter(),
            "d_c": AverageMeter(),
            "d_p": AverageMeter(),
            "rate_term": AverageMeter(),
            "feature_loss": AverageMeter(),
            "iou_fg": AverageMeter(),
            "fg_ratio_gt": AverageMeter(),
            "fg_ratio_pred": AverageMeter(),
        }

        for t in range(T):
            src_feat = src_feats[t]

            # ROI-VFC forward per-frame. When training FST we keep ROI-VFC in no_grad.
            if is_train and optimize_target == "fst":
                with torch.no_grad():
                    roi_out = roi_vfc(src_feat)
            else:
                roi_out = roi_vfc(src_feat)

            feat_rec = roi_out["F_cur_hat"]

            # prepare per-frame image/mask for FST/deeplab
            if has_time:
                img_raw_t = image_raw[:, t]
                mask_t = mask[:, t]
            else:
                img_raw_t = image_raw
                mask_t = mask

            # FST/DeepLab forward: when optimizing codec, keep FST under no_grad
            if is_train and optimize_target == "codec":
                with torch.no_grad():
                    feat_final, recon_image = feature_space_transfer(feat_rec)
                    seg_out = deeplab_adapter(feat_final, return_features=True)
                    logits = F.interpolate(seg_out["out"], size=mask_t.shape[-2:], mode="bilinear", align_corners=False)
                    seg_loss = F.cross_entropy(logits, mask_t)
                    if "aux" in seg_out:
                        aux_logits = F.interpolate(seg_out["aux"], size=mask_t.shape[-2:], mode="bilinear", align_corners=False)
                        seg_loss = seg_loss + aux_weight * F.cross_entropy(aux_logits, mask_t)
            else:
                feat_final, recon_image = feature_space_transfer(feat_rec)
                seg_out = deeplab_adapter(feat_final, return_features=True)
                logits = F.interpolate(seg_out["out"], size=mask_t.shape[-2:], mode="bilinear", align_corners=False)

                seg_loss = F.cross_entropy(logits, mask_t)
                if "aux" in seg_out:
                    aux_logits = F.interpolate(seg_out["aux"], size=mask_t.shape[-2:], mode="bilinear", align_corners=False)
                    seg_loss = seg_loss + aux_weight * F.cross_entropy(aux_logits, mask_t)

            codec_parts = compute_codec_loss(
                roi_out=roi_out,
                weights=codec_weights,
                device=device,
                rate_scale=stage_rate_scale,
                include_rz=stage_include_rz,
            )
            d_mid = F.mse_loss(feat_final, src_feat.detach())

            if recon_image.shape[-2:] != img_raw_t.shape[-2:]:
                target_img = F.interpolate(img_raw_t, size=recon_image.shape[-2:], mode="bilinear", align_corners=False)
            else:
                target_img = img_raw_t
            d_x = F.mse_loss(recon_image, target_img)

            with torch.no_grad():
                src_task = deeplab_adapter(src_feat.detach(), return_features=True)
            src_high = src_task.get("high_features", [])
            tr_high = seg_out.get("high_features", [])
            if len(src_high) > 0 and len(src_high) == len(tr_high):
                d_high = torch.stack(
                    [F.mse_loss(t_h, s_h.detach()) for t_h, s_h in zip(tr_high, src_high)]
                ).mean()
            else:
                d_high = torch.tensor(0.0, device=device)

            fst_parts = compute_fst_loss(
                l_task=seg_loss,
                d_x=d_x,
                d_mid=d_mid,
                d_high=d_high,
                weights=fst_weights,
            )

            if optimize_target == "codec":
                total_loss = codec_parts["l_codec"]
            elif optimize_target == "fst":
                total_loss = fst_parts["l_fst"]
            else:
                total_loss = codec_parts["l_codec"] + fst_parts["l_fst"]

            # accumulate
            batch_meters["total_loss"].update(float(total_loss.item()))
            batch_meters["l_codec"].update(float(codec_parts["l_codec"].item()))
            batch_meters["l_fst"].update(float(fst_parts["l_fst"].item()))
            batch_meters["l_task"].update(float(fst_parts["l_task"].item()))
            batch_meters["d_x"].update(float(fst_parts["d_x"].item()))
            batch_meters["d_mid"].update(float(fst_parts["d_mid"].item()))
            batch_meters["d_high"].update(float(fst_parts["d_high"].item()))
            batch_meters["r_r"].update(float(codec_parts["r_r"].item()))
            batch_meters["r_m"].update(float(codec_parts["r_m"].item()))
            batch_meters["d_f"].update(float(codec_parts["d_f"].item()))
            batch_meters["d_c"].update(float(codec_parts["d_c"].item()))
            batch_meters["d_p"].update(float(codec_parts["d_p"].item()))
            batch_meters["rate_term"].update(float(codec_parts["rate_term"].item()))
            batch_meters["feature_loss"].update(
                float((codec_parts["d_f"] + fst_parts["d_mid"] + fst_parts["d_high"]).item())
            )
            pred = logits.detach().argmax(dim=1)
            batch_meters["iou_fg"].update(float(_compute_miou(logits.detach(), mask_t, num_classes=2)))


            # backprop per-frame (accumulate gradients across frames)
            if is_train and optimizer is not None:
                total_loss.backward()

        # optimizer step once per batch after accumulating gradients from all frames
        if is_train and optimizer is not None:
            optimizer.step()

        batch_metrics = {k: m.avg for k, m in batch_meters.items()}

        for key, meter in totals.items():
            meter.update(float(batch_metrics[key]))
        num_batches += 1

        if comet_exp is not None and log_batch_metrics and (batch_idx % max(1, batch_log_interval) == 0):
            step = (max(epoch, 1) - 1) * total_batches + batch_idx
            comet_exp.log_metrics(
                {
                    f"{split}/batch_total_loss": float(batch_metrics["total_loss"]),
                    f"{split}/batch_l_codec": float(batch_metrics["l_codec"]),
                    f"{split}/batch_l_fst": float(batch_metrics["l_fst"]),
                    f"{split}/batch_rate_term": float(batch_metrics["rate_term"]),
                    f"{split}/batch_feature_loss": float(batch_metrics["feature_loss"]),
                    f"{split}/batch_roi_feature_loss_df": float(batch_metrics["d_f"]),
                    f"{split}/batch_fst_feature_loss_dmid": float(batch_metrics["d_mid"]),
                    f"{split}/batch_fst_feature_loss_dhigh": float(batch_metrics["d_high"]),
                    f"{split}/batch_image_loss_dx": float(batch_metrics["d_x"]),
                    f"{split}/batch_iou_fg": float(batch_metrics["iou_fg"]),

                },
                epoch=epoch,
                step=step,
            )

        if use_tqdm and hasattr(iterator, "set_postfix_str"):
            if optimize_target == "codec":
                iterator.set_postfix_str(
                    f"opt=codec | "
                    f"codec={batch_metrics['l_codec']:.3f} | "
                    f"rate={batch_metrics['rate_term']:.3f} | "
                    f"roi_df={batch_metrics['d_f']:.3f}"
                )

            elif optimize_target == "fst":
                iterator.set_postfix_str(
                    f"opt=fst | "
                    f"fst={batch_metrics['l_fst']:.3f} | "
                    f"dx={batch_metrics['d_x']:.3f} | "
                    f"fst_mid={batch_metrics['d_mid']:.3f} | "
                    f"fst_high={batch_metrics['d_high']:.3f} | "
                    f"iou={batch_metrics['iou_fg']:.3f}"
                )

            else:
                iterator.set_postfix_str(
                    f"opt=both | "
                    f"codec={batch_metrics['l_codec']:.3f} | "
                    f"fst={batch_metrics['l_fst']:.3f} | "
                    f"rate={batch_metrics['rate_term']:.3f} | "
                    f"roi_df={batch_metrics['d_f']:.3f} | "
                    f"fst_mid={batch_metrics['d_mid']:.3f} | "
                    f"fst_high={batch_metrics['d_high']:.3f} | "
                    f"dx={batch_metrics['d_x']:.3f} | "
                    f"iou={batch_metrics['iou_fg']:.3f}"
                )

    if num_batches == 0:
        return {k: 0.0 for k in totals}
    return {k: m.avg for k, m in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train ROI-VFC + FST with frozen DeepLabV3 on YTVOS/VOS")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ytvos/roi_vfc_deeplabv3_resnet50.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--valid-batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable-comet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = load_config(str(cfg_path)) or {}

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})
    logging_cfg = cfg.get("logging", {})

    train_bs = int(_resolve_setting(args.train_batch_size, training_cfg.get("train_batch_size"), 8))
    val_bs = int(_resolve_setting(args.valid_batch_size, training_cfg.get("valid_batch_size"), 1))
    num_workers = int(training_cfg.get("num_workers", 4))
    num_classes = int(model_cfg.get("data", {}).get("num_classes", 124))
    device = _resolve_device(_resolve_setting(args.device, training_cfg.get("device"), "auto"))

    if args.max_train_samples is not None:
        cfg.setdefault("data", {})["max_train_samples"] = int(args.max_train_samples)
    if args.max_valid_samples is not None:
        cfg.setdefault("data", {})["max_valid_samples"] = int(args.max_valid_samples)
    if args.height is not None or args.width is not None:
        h0, w0 = cfg.setdefault("data", {}).get("image_size", [384, 640])
        cfg["data"]["image_size"] = [
            int(args.height if args.height is not None else h0),
            int(args.width if args.width is not None else w0),
        ]
    if args.stage1_epochs is not None or args.stage2_epochs is not None:
        stages = cfg.setdefault("training", {}).setdefault("stages", [])
        while len(stages) < 2:
            stages.append(
                {
                    "name": f"stage{len(stages)+1}",
                    "optimize": "codec" if len(stages) == 0 else "fst",
                    "epochs": 1,
                    "lr": float(training_cfg.get("lr", 1e-4)),
                    "weight_decay": float(training_cfg.get("weight_decay", 1e-4)),
                }
            )
        if args.stage1_epochs is not None:
            stages[0]["epochs"] = max(1, int(args.stage1_epochs))
        if args.stage2_epochs is not None:
            stages[1]["epochs"] = max(1, int(args.stage2_epochs))
    if args.disable_comet:
        cfg.setdefault("logging", {}).setdefault("comet", {})["enabled"] = False

    save_dir = Path(
        _resolve_setting(
            args.save_dir,
            output_cfg.get("save_dir"),
            f"outputs/{cfg.get('experiment_name', 'roi_vfc_deeplabv3_ytvos')}",
        )
    )
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _build_dataloaders(cfg, num_workers, train_bs, val_bs, seed)

    roi_cfg = model_cfg.get("roi_vfc", {})
    roi_vfc = ROI_VFC(
        img_channels=int(roi_cfg.get("img_channels", 3)),
        feat_channels=int(roi_cfg.get("feat_channels", 64)),
        y_chan=int(roi_cfg.get("y_chan", 96)),
        lambda_rd=float(roi_cfg.get("lambda_rd", 0.01)),
    ).to(device)

    feat_cfg = model_cfg.get("feature_extraction", {})
    perc_cfg = model_cfg.get("perception_extraction", {})

    roi_vfc.feat_extraction = FeatureExtraction(
        backbone_name=str(feat_cfg.get("backbone", "resnet50")),
        pretrained=bool(feat_cfg.get("pretrained", True)),
    ).to(device)
    roi_vfc.perception_extraction = PerceptionExtraction(
        backbone_name=str(perc_cfg.get("backbone", "resnet50")),
        pretrained=bool(perc_cfg.get("pretrained", True)),
        remove_res1=bool(perc_cfg.get("remove_res1", True)),
    ).to(device)

    if bool(feat_cfg.get("freeze", True)):
        _freeze_module(roi_vfc.feat_extraction)
    if bool(perc_cfg.get("freeze", True)):
        _freeze_module(roi_vfc.perception_extraction)

    fst_cfg = model_cfg.get("feature_space_transfer", {})
    feature_space_transfer = FeatureSpaceTransfer(
        in_channel=int(fst_cfg.get("in_channel", 256)),
        mid_channel=int(fst_cfg.get("mid_channel", 64)),
    ).to(device)

    deeplab_cfg = model_cfg.get("deeplabv3", {})
    deeplab_num_classes = int(deeplab_cfg.get("num_classes", 2))
    deeplab_weights_path = deeplab_cfg.get("weights_path")
    deeplab_adapter = _build_deeplab_feature_adapter(
        pretrained=bool(deeplab_cfg.get("pretrained", True)),
        device=device,
        num_classes=num_classes,
        weights_path=Path(deeplab_weights_path) if deeplab_weights_path else None,
    )
    if bool(deeplab_cfg.get("freeze", True)):
        _freeze_module(deeplab_adapter)

    aux_weight = float(deeplab_cfg.get("aux_weight", 0.4))

    lw_cfg = cfg.get("loss_weights", {})
    codec_cfg = lw_cfg.get("codec", {}) if isinstance(lw_cfg, dict) else {}
    fst_cfg = lw_cfg.get("fst", {}) if isinstance(lw_cfg, dict) else {}
    codec_weights = {
        "lambda_r": float(codec_cfg.get("lambda_r", 1.0)),
        "lambda_f": float(codec_cfg.get("lambda_f", 1.0)),
        "lambda_c": float(codec_cfg.get("lambda_c", 1.0)),
        "lambda_p": float(codec_cfg.get("lambda_p", 1.0)),
    }
    fst_weights = {
        "lambda_task": float(fst_cfg.get("lambda_task", 10.0)),
        "lambda_x": float(fst_cfg.get("lambda_x", 1024.0)),
        "lambda_mid": float(fst_cfg.get("lambda_mid", 16.0)),
        "lambda_high": float(fst_cfg.get("lambda_high", 64.0)),
    }

    training_stages = _build_training_stages(training_cfg)
    total_epochs = sum(int(s.get("epochs", 0)) for s in training_stages)

    if args.epochs is not None and args.epochs > 0:
        # Keep CLI compatibility: scale down from the beginning of stage list.
        remaining = int(args.epochs)
        adjusted: List[Dict[str, Any]] = []
        for stage in training_stages:
            if remaining <= 0:
                break
            s = dict(stage)
            s_epochs = min(int(s["epochs"]), remaining)
            s["epochs"] = s_epochs
            adjusted.append(s)
            remaining -= s_epochs
        if len(adjusted) > 0:
            training_stages = adjusted
            total_epochs = sum(int(s.get("epochs", 0)) for s in training_stages)

    roi_base_requires_grad = {name: p.requires_grad for name, p in roi_vfc.named_parameters()}
    fst_base_requires_grad = {name: p.requires_grad for name, p in feature_space_transfer.named_parameters()}

    run_name = str(cfg.get("experiment_name", "roi_vfc_deeplabv3_ytvos"))

    resolved_config = {
        "config_path": str(cfg_path),
        "experiment_name": run_name,
        "seed": seed,
        "device": str(device),
        "epochs": total_epochs,
        "training_stages": training_stages,
        "train_batch_size": train_bs,
        "valid_batch_size": val_bs,
        "num_workers": num_workers,
        "aux_weight": aux_weight,
        "codec_weights": codec_weights,
        "fst_weights": fst_weights,
        "data": cfg.get("data", {}),
        "model": model_cfg,
    }

    print(
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"device={device} train_bs={train_bs} val_bs={val_bs}"
    )
    print("Freeze modules: feature_extraction=True perception_extraction=True deeplabv3=True")

    use_tqdm = bool(logging_cfg.get("use_tqdm", True))
    log_batch_metrics = bool(logging_cfg.get("log_batch_metrics", True))
    batch_log_interval = int(logging_cfg.get("batch_log_interval", 1))
    auto_load_best_for_fst = bool(training_cfg.get("auto_load_best_for_fst", False))
    history = []
    best_val = float("inf")
    resume_ckpt = None
    resume_stage_idx = 1
    resume_stage_optimize = None
    resume_optimizer_state = None
    start_stage_idx = 1
    start_local_epoch = 1
    global_epoch = 0

    resolved_config["logging"] = {
        "use_tqdm": use_tqdm,
        "log_batch_metrics": log_batch_metrics,
        "batch_log_interval": batch_log_interval,
    }

    resume_path = _resolve_setting(args.resume, training_cfg.get("resume_from"), None)
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        resume_ckpt = torch.load(str(resume_path), map_location=device)
        roi_vfc.load_state_dict(resume_ckpt.get("roi_vfc_state", {}), strict=False)
        feature_space_transfer.load_state_dict(
            resume_ckpt.get("feature_space_transfer_state", {}), strict=False
        )

        best_val = float(resume_ckpt.get("best_val", best_val))
        history = resume_ckpt.get("history", history)
        resume_stage_idx = int(resume_ckpt.get("stage_index", 1))
        resume_stage_optimize = resume_ckpt.get("stage_optimize")
        resume_optimizer_state = resume_ckpt.get("optimizer_state")

        resume_comet_key = resume_ckpt.get("comet_experiment_key")
        if resume_comet_key:
            logging_cfg.setdefault("comet", {})["experiment_key"] = resume_comet_key

        resume_global_epoch = int(resume_ckpt.get("epoch", 0))
        next_global_epoch = resume_global_epoch + 1
        if next_global_epoch > total_epochs:
            print(
                f"Resume checkpoint epoch {resume_global_epoch} already covers all epochs. "
                "Nothing to resume."
            )
            return

        if "stage_epoch" in resume_ckpt:
            start_stage_idx = resume_stage_idx
            start_local_epoch = int(resume_ckpt.get("stage_epoch", 0)) + 1
        else:
            loc = _locate_stage_for_epoch(training_stages, next_global_epoch)
            start_stage_idx = int(loc["stage_index"])
            start_local_epoch = int(loc["local_epoch"])

        global_epoch = next_global_epoch - 1

    baseline_metrics = _validate_deeplab_baseline(
        dataloader=val_loader,
        feature_extractor=roi_vfc.feat_extraction,
        deeplab_adapter=deeplab_adapter,
        device=device,
        aux_weight=aux_weight,
        use_tqdm=use_tqdm,
        num_classes=num_classes,
    )
    print(
        "DeepLab baseline (src features) on val: "
        f"iou_fg={baseline_metrics['iou_fg']:.4f} "
        f"l_task={baseline_metrics['l_task']:.4f} "

    )

    comet_exp = _build_comet_experiment(logging_cfg, run_name)
    if comet_exp is not None:
        resolved_config.setdefault("comet", {})["experiment_key"] = comet_exp.get_key()
        comet_exp.log_parameters(_flatten_dict(resolved_config))
    for stage_idx, stage in enumerate(training_stages, start=1):
        stage_name = str(stage.get("name", f"stage{stage_idx}"))
        stage_optimize = str(stage.get("optimize", "codec")).lower()
        stage_epochs = max(1, int(stage.get("epochs", 1)))
        stage_lr = float(stage.get("lr", training_cfg.get("lr", 1e-4)))
        stage_wd = float(stage.get("weight_decay", training_cfg.get("weight_decay", 1e-4)))
        stage_codec = stage.get("codec", {}) if isinstance(stage.get("codec", {}), dict) else {}
        stage_rate_scale = float(stage_codec.get("rate_scale", 1.0))
        stage_include_rz = bool(stage_codec.get("include_rz", True))

        if stage_idx < start_stage_idx:
            continue

        if auto_load_best_for_fst and stage_optimize == "fst" and resume_ckpt is None:
            best_path = save_dir / "best.pt"
            if best_path.exists():
                best_ckpt = torch.load(str(best_path), map_location=device)
                roi_vfc.load_state_dict(best_ckpt.get("roi_vfc_state", {}), strict=False)
                print(f"Loaded best checkpoint before FST stage: {best_path}")

        _set_module_trainability(roi_vfc, roi_base_requires_grad, enabled=stage_optimize in {"codec", "both"})
        _set_module_trainability(
            feature_space_transfer,
            fst_base_requires_grad,
            enabled=stage_optimize in {"fst", "both"},
        )

        roi_trainable_params = sum(p.numel() for p in roi_vfc.parameters() if p.requires_grad)
        fst_trainable_params = sum(p.numel() for p in feature_space_transfer.parameters() if p.requires_grad)

        optimizer = _build_optimizer(
            roi_vfc=roi_vfc,
            feature_space_transfer=feature_space_transfer,
            stage_optimize=stage_optimize,
            lr=stage_lr,
            weight_decay=stage_wd,
        )

        if (
            resume_ckpt is not None
            and stage_idx == resume_stage_idx
            and resume_stage_optimize == stage_optimize
            and resume_optimizer_state is not None
        ):
            optimizer.load_state_dict(resume_optimizer_state)

        print(
            f"\n=== Stage {stage_idx}/{len(training_stages)}: {stage_name} "
            f"(optimize={stage_optimize}, epochs={stage_epochs}, lr={stage_lr:.2e}, "
            f"rate_scale={stage_rate_scale}, include_rz={stage_include_rz}) ==="
        )
        print(
            "trainable params: "
            f"roi_vfc={roi_trainable_params} "
            f"fst={fst_trainable_params}"
        )

        local_start = start_local_epoch if stage_idx == start_stage_idx else 1
        for local_epoch in range(local_start, stage_epochs + 1):
            global_epoch += 1
            print(
                f"\nEpoch {global_epoch}/{total_epochs} "
                f"(stage {stage_idx} epoch {local_epoch}/{stage_epochs})"
            )

            train_metrics = _run_epoch(
                mode="train",
                dataloader=train_loader,
                roi_vfc=roi_vfc,
                feature_extractor=roi_vfc.feat_extraction,
                feature_space_transfer=feature_space_transfer,
                deeplab_adapter=deeplab_adapter,
                optimizer=optimizer,
                device=device,
                aux_weight=aux_weight,
                codec_weights=codec_weights,
                fst_weights=fst_weights,
                use_tqdm=use_tqdm,
                comet_exp=comet_exp,
                epoch=global_epoch,
                split="train",
                log_batch_metrics=log_batch_metrics,
                batch_log_interval=batch_log_interval,
                optimize_target=stage_optimize,
                codec_stage_cfg=stage_codec,
            )
            with torch.no_grad():
                val_metrics = _run_epoch(
                    mode="val",
                    dataloader=val_loader,
                    roi_vfc=roi_vfc,
                    feature_extractor=roi_vfc.feat_extraction,
                    feature_space_transfer=feature_space_transfer,
                    deeplab_adapter=deeplab_adapter,
                    optimizer=None,
                    device=device,
                    aux_weight=aux_weight,
                    codec_weights=codec_weights,
                    fst_weights=fst_weights,
                    use_tqdm=use_tqdm,
                    comet_exp=comet_exp,
                    epoch=global_epoch,
                    split="val",
                    log_batch_metrics=log_batch_metrics,
                    batch_log_interval=batch_log_interval,
                    optimize_target=stage_optimize,
                    codec_stage_cfg=stage_codec,
                )

            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_index": stage_idx,
                "stage_optimize": stage_optimize,
                "stage_codec": {
                    "rate_scale": stage_rate_scale,
                    "include_rz": stage_include_rz,
                },
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)

            is_best = val_metrics["total_loss"] < best_val
            if is_best:
                best_val = val_metrics["total_loss"]

            print(
                "train: "
                f"total={train_metrics['total_loss']:.4f} "
                f"codec={train_metrics['l_codec']:.4f} fst={train_metrics['l_fst']:.4f} "
                f"rd={train_metrics['rate_term']:.4f} feature={train_metrics['feature_loss']:.4f} "
                f"(d_f={train_metrics['d_f']:.4f}, d_mid={train_metrics['d_mid']:.4f}, d_high={train_metrics['d_high']:.4f}) "
                f"fg_gt={train_metrics['fg_ratio_gt']:.4f} fg_pred={train_metrics['fg_ratio_pred']:.4f}"
            )
            print(
                "val:   "
                f"total={val_metrics['total_loss']:.4f} "
                f"codec={val_metrics['l_codec']:.4f} fst={val_metrics['l_fst']:.4f} "
                f"rd={val_metrics['rate_term']:.4f} feature={val_metrics['feature_loss']:.4f} "
                f"iou_fg={val_metrics['iou_fg']:.4f} "
                f"fg_gt={val_metrics['fg_ratio_gt']:.4f} fg_pred={val_metrics['fg_ratio_pred']:.4f} "
                f"(d_f={val_metrics['d_f']:.4f}, d_mid={val_metrics['d_mid']:.4f}, d_high={val_metrics['d_high']:.4f})"
            )

            if comet_exp is not None:
                comet_exp.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, epoch=global_epoch)
                comet_exp.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, epoch=global_epoch)
                comet_exp.log_metrics(
                    {
                        "train/stage_idx": stage_idx,
                        "train/stage_optimize": 0 if stage_optimize == "codec" else (1 if stage_optimize == "fst" else 2),
                    },
                    epoch=global_epoch,
                )

            ckpt = {
                "epoch": global_epoch,
                "stage": row["stage"],
                "stage_index": row["stage_index"],
                "stage_epoch": local_epoch,
                "stage_optimize": row["stage_optimize"],
                "stage_codec": row["stage_codec"],
                "comet_experiment_key": (comet_exp.get_key() if comet_exp is not None else None),
                "roi_vfc_state": roi_vfc.state_dict(),
                "feature_space_transfer_state": feature_space_transfer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
                "history": history,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": resolved_config,
            }

            torch.save(ckpt, save_dir / f"epoch_{global_epoch:03d}.pt")
            torch.save(ckpt, save_dir / "last.pt")

            if is_best:
                torch.save(ckpt, save_dir / "best.pt")
                print('saved best checkpoint')

            _append_metrics(save_dir / "metrics.json", history)

    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if comet_exp is not None:
        comet_exp.log_asset(str(save_dir / "metrics.json"))
        comet_exp.end()

    print(f"Training done. Outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
