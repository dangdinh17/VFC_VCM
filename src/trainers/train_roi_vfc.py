"""Training utilities for ROI-VFC + FeatureSpaceTransfer + downstream tasks.

Pipeline:
1) frames -> FeatureExtraction -> source features
2) source features -> ROI_VFC -> compressed/reconstructed features
3) reconstructed features -> FeatureSpaceTransfer -> transferred features + reconstructed image
4) transferred features -> downstream models (detection/segmentation)
5) optimize with ROI rate-distortion + feature transfer + downstream consistency losses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class LossWeights:
    roi_rate: float = 1.0
    roi_reconstruction: float = 1.0
    transfer_domain: float = 1.0
    transfer_reconstruction: float = 1.0
    downstream_supervised: float = 1.0
    downstream_consistency: float = 1.0


def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(_to_device(x, device) for x in obj)
    return obj


def _extract_first_tensor(output: Any) -> Optional[Tensor]:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for v in output.values():
            t = _extract_first_tensor(v)
            if t is not None:
                return t
    if isinstance(output, (list, tuple)):
        for v in output:
            t = _extract_first_tensor(v)
            if t is not None:
                return t
    return None


def compute_roi_vfc_losses(
    roi_output: Mapping[str, Any],
    source_features: Tensor,
) -> Dict[str, Tensor]:
    """Compute rate + reconstruction losses for ROI-VFC."""
    device = source_features.device
    compression_loss = roi_output.get("bpp", torch.tensor(0.0, device=device))
    if not torch.is_tensor(compression_loss):
        compression_loss = torch.tensor(float(compression_loss), device=device)

    f_hat = roi_output["F_cur_hat"]
    recon_core = F.mse_loss(f_hat, source_features)

    mse_c = roi_output.get("mse_c", torch.tensor(0.0, device=device))
    mse_p = roi_output.get("mse_p", torch.tensor(0.0, device=device))
    if not torch.is_tensor(mse_c):
        mse_c = torch.tensor(float(mse_c), device=device)
    if not torch.is_tensor(mse_p):
        mse_p = torch.tensor(float(mse_p), device=device)

    reconstruction_loss = recon_core + mse_c + mse_p

    return {
        "roi_rate_loss": compression_loss,
        "roi_reconstruction_loss": reconstruction_loss,
    }


def _prepare_batch_labels(
    labels_batch: Any,
    target_features_batch: Any,
    device: torch.device,
) -> Dict[str, Any]:
    labels = labels_batch if isinstance(labels_batch, list) else []

    stacked_target_features = None
    if isinstance(target_features_batch, list) and len(target_features_batch) > 0:
        if all(torch.is_tensor(x) for x in target_features_batch):
            stacked_target_features = torch.stack(target_features_batch, dim=0).to(device)

    seg_masks = []
    for item in labels:
        if isinstance(item, dict) and torch.is_tensor(item.get("segmentation_mask")):
            seg_masks.append(item["segmentation_mask"])

    if len(seg_masks) > 0 and len(seg_masks) == len(labels):
        seg_masks = torch.stack(seg_masks, dim=0).to(device)
    else:
        seg_masks = None

    return {
        "labels": labels,
        "target_features": stacked_target_features,
        "segmentation_masks": seg_masks,
    }


def _safe_reset_roi_buffer(roi_vfc: torch.nn.Module) -> None:
    if hasattr(roi_vfc, "reset_buffer") and callable(getattr(roi_vfc, "reset_buffer")):
        roi_vfc.reset_buffer()


def _normalize_temporal_labels(
    labels_seq: Any,
    t: int,
) -> Dict[str, Any]:
    if isinstance(labels_seq, list) and len(labels_seq) > t and isinstance(labels_seq[t], dict):
        return labels_seq[t]
    return {}


def _normalize_temporal_target(
    target_seq: Any,
    t: int,
) -> Optional[Tensor]:
    if isinstance(target_seq, list) and len(target_seq) > t:
        val = target_seq[t]
        if torch.is_tensor(val):
            return val
    if torch.is_tensor(target_seq):
        if target_seq.dim() >= 1 and target_seq.size(0) > t:
            return target_seq[t]
        return target_seq
    return None


def _run_temporal_step(
    roi_vfc: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    feature_space_transfer: torch.nn.Module,
    frame_t: Tensor,
    labels_t: Dict[str, Any],
    target_t: Optional[Tensor],
    downstream_models: Mapping[str, torch.nn.Module],
    task_criteria: Optional[Mapping[str, Callable[[Any, Any], Tensor]]],
    loss_weights: LossWeights,
) -> Dict[str, Tensor]:
    """Run one temporal step and return all scalar losses."""
    with torch.no_grad():
        source_features = feature_extractor(frame_t)

    roi_output = roi_vfc(source_features)
    transfer_output = feature_space_transfer(roi_output["F_cur_hat"])
    if isinstance(transfer_output, (tuple, list)):
        transferred_features = transfer_output[0]
        reconstructed_image = transfer_output[1] if len(transfer_output) > 1 else None
    else:
        transferred_features = transfer_output
        reconstructed_image = None

    target_pack: List[Optional[Tensor]]
    if target_t is None:
        target_pack = [None]
    else:
        target_pack = [target_t]

    labels_pack = _prepare_batch_labels([labels_t], target_pack, frame_t.device)
    roi_losses = compute_roi_vfc_losses(roi_output=roi_output, source_features=source_features)
    transfer_losses = compute_feature_transfer_losses(
        transferred_features=transferred_features,
        reconstructed_image=reconstructed_image,
        source_features=source_features,
        input_frames=frame_t,
        downstream_models=downstream_models,
        batch_labels=labels_pack,
        task_criteria=task_criteria,
    )

    total = (
        loss_weights.roi_rate * roi_losses["roi_rate_loss"]
        + loss_weights.roi_reconstruction * roi_losses["roi_reconstruction_loss"]
        + loss_weights.transfer_domain * transfer_losses["transfer_domain_loss"]
        + loss_weights.transfer_reconstruction * transfer_losses["transfer_reconstruction_loss"]
        + loss_weights.downstream_supervised * transfer_losses["downstream_supervised_loss"]
        + loss_weights.downstream_consistency * transfer_losses["downstream_consistency_loss"]
    )

    return {
        "total_loss": total,
        "roi_rate_loss": roi_losses["roi_rate_loss"],
        "roi_reconstruction_loss": roi_losses["roi_reconstruction_loss"],
        "transfer_domain_loss": transfer_losses["transfer_domain_loss"],
        "transfer_reconstruction_loss": transfer_losses["transfer_reconstruction_loss"],
        "downstream_supervised_loss": transfer_losses["downstream_supervised_loss"],
        "downstream_consistency_loss": transfer_losses["downstream_consistency_loss"],
    }


def compute_feature_transfer_losses(
    transferred_features: Tensor,
    reconstructed_image: Optional[Tensor],
    source_features: Tensor,
    input_frames: Tensor,
    downstream_models: Optional[Mapping[str, torch.nn.Module]] = None,
    batch_labels: Optional[Dict[str, Any]] = None,
    task_criteria: Optional[Mapping[str, Callable[[Any, Any], Tensor]]] = None,
) -> Dict[str, Tensor]:
    """Compute feature-space transfer and downstream related losses."""
    device = source_features.device
    downstream_models = downstream_models or {}
    task_criteria = task_criteria or {}

    target_features = None
    segmentation_masks = None
    labels = []
    if batch_labels is not None:
        target_features = batch_labels.get("target_features")
        segmentation_masks = batch_labels.get("segmentation_masks")
        labels = batch_labels.get("labels", [])

    # Source -> target domain transfer loss
    if torch.is_tensor(target_features):
        domain_target = target_features
        if domain_target.shape != transferred_features.shape:
            domain_target = F.interpolate(
                domain_target,
                size=transferred_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
    else:
        domain_target = source_features.detach()
    transfer_domain_loss = F.mse_loss(transferred_features, domain_target)

    # Image reconstruction loss from transfer module
    transfer_reconstruction_loss = torch.tensor(0.0, device=device)
    if torch.is_tensor(reconstructed_image):
        current_frame = input_frames[:, -1] if input_frames.dim() == 5 else input_frames
        if current_frame.dtype != torch.float32:
            current_frame = current_frame.float()
        if current_frame.max() > 1.5:
            current_frame = current_frame / 255.0
        if reconstructed_image.shape[-2:] != current_frame.shape[-2:]:
            current_frame = F.interpolate(
                current_frame,
                size=reconstructed_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        transfer_reconstruction_loss = F.mse_loss(reconstructed_image, current_frame)

    downstream_supervised_loss = torch.tensor(0.0, device=device)
    downstream_consistency_loss = torch.tensor(0.0, device=device)

    # Downstream consistency + optional supervised losses
    for task_name, model in downstream_models.items():
        pred_source = model(source_features.detach())
        pred_transferred = model(transferred_features)

        source_tensor = _extract_first_tensor(pred_source)
        transferred_tensor = _extract_first_tensor(pred_transferred)
        if source_tensor is not None and transferred_tensor is not None:
            downstream_consistency_loss = downstream_consistency_loss + F.mse_loss(
                transferred_tensor,
                source_tensor.detach(),
            )

        criterion = task_criteria.get(task_name)
        if criterion is not None:
            target = None
            if task_name == "segmentation" and torch.is_tensor(segmentation_masks):
                target = segmentation_masks
            elif len(labels) > 0:
                task_targets = [x.get(task_name) if isinstance(x, dict) else None for x in labels]
                if all(torch.is_tensor(t) for t in task_targets):
                    target = torch.stack(task_targets, dim=0).to(device)
            if target is not None:
                downstream_supervised_loss = downstream_supervised_loss + criterion(pred_transferred, target)

    return {
        "transfer_domain_loss": transfer_domain_loss,
        "transfer_reconstruction_loss": transfer_reconstruction_loss,
        "downstream_supervised_loss": downstream_supervised_loss,
        "downstream_consistency_loss": downstream_consistency_loss,
    }


def train_roi_vfc_epoch(
    roi_vfc: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    feature_space_transfer: torch.nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    downstream_models: Optional[Mapping[str, torch.nn.Module]] = None,
    task_criteria: Optional[Mapping[str, Callable[[Any, Any], Tensor]]] = None,
    loss_weights: Optional[LossWeights] = None,
) -> Dict[str, float]:
    """Train ROI-VFC pipeline for one epoch."""
    if device is None:
        device = next(roi_vfc.parameters()).device
    loss_weights = loss_weights or LossWeights()
    downstream_models = downstream_models or {}

    roi_vfc.train()
    feature_space_transfer.train()
    feature_extractor.eval()
    for model in downstream_models.values():
        model.train()

    total: Dict[str, float] = {
        "total_loss": 0.0,
        "roi_rate_loss": 0.0,
        "roi_reconstruction_loss": 0.0,
        "transfer_domain_loss": 0.0,
        "transfer_reconstruction_loss": 0.0,
        "downstream_supervised_loss": 0.0,
        "downstream_consistency_loss": 0.0,
    }

    num_steps = 0

    for batch in dataloader:
        batch = _to_device(batch, device)
        frames = batch["frames"]  # [B, T, C, H, W]
        frame_mask = batch.get("frame_mask")
        labels_seq_batch = batch.get("labels", [])
        target_seq_batch = batch.get("target_features", [])

        if frame_mask is None:
            frame_mask = torch.ones(frames.size(0), frames.size(1), dtype=torch.bool, device=frames.device)

        optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.tensor(0.0, device=device)
        batch_steps = 0

        # Train temporally for each sample to preserve reference chain strictly.
        for b in range(frames.size(0)):
            _safe_reset_roi_buffer(roi_vfc)
            labels_seq = labels_seq_batch[b] if isinstance(labels_seq_batch, list) and len(labels_seq_batch) > b else []
            target_seq = target_seq_batch[b] if isinstance(target_seq_batch, list) and len(target_seq_batch) > b else []

            for t in range(frames.size(1)):
                if not bool(frame_mask[b, t].item()):
                    continue

                frame_t = frames[b : b + 1, t]
                labels_t = _normalize_temporal_labels(labels_seq, t)
                target_t = _normalize_temporal_target(target_seq, t)

                losses = _run_temporal_step(
                    roi_vfc=roi_vfc,
                    feature_extractor=feature_extractor,
                    feature_space_transfer=feature_space_transfer,
                    frame_t=frame_t,
                    labels_t=labels_t,
                    target_t=target_t,
                    downstream_models=downstream_models,
                    task_criteria=task_criteria,
                    loss_weights=loss_weights,
                )

                batch_loss = batch_loss + losses["total_loss"]
                batch_steps += 1
                num_steps += 1

                total["total_loss"] += float(losses["total_loss"].item())
                total["roi_rate_loss"] += float(losses["roi_rate_loss"].item())
                total["roi_reconstruction_loss"] += float(losses["roi_reconstruction_loss"].item())
                total["transfer_domain_loss"] += float(losses["transfer_domain_loss"].item())
                total["transfer_reconstruction_loss"] += float(losses["transfer_reconstruction_loss"].item())
                total["downstream_supervised_loss"] += float(losses["downstream_supervised_loss"].item())
                total["downstream_consistency_loss"] += float(losses["downstream_consistency_loss"].item())

        if batch_steps == 0:
            continue

        (batch_loss / float(batch_steps)).backward()
        optimizer.step()

    if num_steps == 0:
        return total

    return {k: v / num_steps for k, v in total.items()}


def test_roi_vfc_epoch(
    roi_vfc: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    feature_space_transfer: torch.nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: Optional[torch.device] = None,
    downstream_models: Optional[Mapping[str, torch.nn.Module]] = None,
    task_criteria: Optional[Mapping[str, Callable[[Any, Any], Tensor]]] = None,
    loss_weights: Optional[LossWeights] = None,
) -> Dict[str, float]:
    """Validation/testing for one epoch with same loss terms as training."""
    if device is None:
        device = next(roi_vfc.parameters()).device
    loss_weights = loss_weights or LossWeights()
    downstream_models = downstream_models or {}

    roi_vfc.eval()
    feature_space_transfer.eval()
    feature_extractor.eval()
    for model in downstream_models.values():
        model.eval()

    total: Dict[str, float] = {
        "total_loss": 0.0,
        "roi_rate_loss": 0.0,
        "roi_reconstruction_loss": 0.0,
        "transfer_domain_loss": 0.0,
        "transfer_reconstruction_loss": 0.0,
        "downstream_supervised_loss": 0.0,
        "downstream_consistency_loss": 0.0,
    }

    num_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _to_device(batch, device)
            frames = batch["frames"]  # [B, T, C, H, W]
            frame_mask = batch.get("frame_mask")
            labels_seq_batch = batch.get("labels", [])
            target_seq_batch = batch.get("target_features", [])

            if frame_mask is None:
                frame_mask = torch.ones(frames.size(0), frames.size(1), dtype=torch.bool, device=frames.device)

            for b in range(frames.size(0)):
                _safe_reset_roi_buffer(roi_vfc)
                labels_seq = labels_seq_batch[b] if isinstance(labels_seq_batch, list) and len(labels_seq_batch) > b else []
                target_seq = target_seq_batch[b] if isinstance(target_seq_batch, list) and len(target_seq_batch) > b else []

                for t in range(frames.size(1)):
                    if not bool(frame_mask[b, t].item()):
                        continue

                    frame_t = frames[b : b + 1, t]
                    labels_t = _normalize_temporal_labels(labels_seq, t)
                    target_t = _normalize_temporal_target(target_seq, t)

                    losses = _run_temporal_step(
                        roi_vfc=roi_vfc,
                        feature_extractor=feature_extractor,
                        feature_space_transfer=feature_space_transfer,
                        frame_t=frame_t,
                        labels_t=labels_t,
                        target_t=target_t,
                        downstream_models=downstream_models,
                        task_criteria=task_criteria,
                        loss_weights=loss_weights,
                    )

                    num_steps += 1
                    total["total_loss"] += float(losses["total_loss"].item())
                    total["roi_rate_loss"] += float(losses["roi_rate_loss"].item())
                    total["roi_reconstruction_loss"] += float(losses["roi_reconstruction_loss"].item())
                    total["transfer_domain_loss"] += float(losses["transfer_domain_loss"].item())
                    total["transfer_reconstruction_loss"] += float(losses["transfer_reconstruction_loss"].item())
                    total["downstream_supervised_loss"] += float(losses["downstream_supervised_loss"].item())
                    total["downstream_consistency_loss"] += float(losses["downstream_consistency_loss"].item())

    if num_steps == 0:
        return total

    return {k: v / num_steps for k, v in total.items()}
