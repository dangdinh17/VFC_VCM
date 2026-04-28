"""Test dataset for ROI-VFC video feature codec pipeline.

Deterministic sampling version for evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from .train_dataset import (
    _align_labels_for_indices,
    _align_target_features_for_indices,
    _load_manifest,
    _read_image,
    _read_video_frames,
    _resolve_total_frames,
)


class TestVideoFeatureDataset(Dataset):
    """Deterministic dataset for ROI-VFC testing and downstream evaluation."""

    def __init__(
        self,
        manifest: Union[str, Path, List[Dict[str, Any]]],
        clip_length: int = 2,
        image_size: Optional[Sequence[int]] = (256, 256),
    ) -> None:
        super().__init__()
        self.samples = _load_manifest(manifest)
        self.clip_length = int(clip_length)
        self.image_size = image_size

        if self.clip_length < 1:
            raise ValueError("clip_length must be >= 1")

    def __len__(self) -> int:
        return len(self.samples)

    def _deterministic_indices(self, num_frames: int) -> List[int]:
        if num_frames <= self.clip_length:
            return [min(i, num_frames - 1) for i in range(self.clip_length)]
        # Use center clip for stable evaluation
        start = (num_frames - self.clip_length) // 2
        return list(range(start, start + self.clip_length))

    def _read_clip(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, List[int]]:
        if "frames" in sample:
            frame_paths = sample["frames"]
            if len(frame_paths) == 0:
                raise ValueError("Sample has empty 'frames' list")
            indices = self._deterministic_indices(len(frame_paths))
            frames = [_read_image(frame_paths[i], self.image_size) for i in indices]
            return torch.stack(frames, dim=0), indices

        if "video" in sample:
            num_frames = _resolve_total_frames(sample)
            indices = self._deterministic_indices(max(num_frames, 1))
            return _read_video_frames(sample["video"], indices, self.image_size), indices

        raise ValueError("Each sample must contain either 'frames' or 'video'.")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        clip, frame_indices = self._read_clip(sample)
        length = int(clip.size(0))

        labels = _align_labels_for_indices(sample, frame_indices, self.image_size)
        target_features = _align_target_features_for_indices(sample, frame_indices)

        return {
            "frames": clip,
            "frame_indices": frame_indices,
            "length": length,
            "labels": labels,
            "target_features": target_features,
            "video_id": sample.get("video_id", f"sample_{index}"),
        }


def test_video_feature_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_t = max(int(item["length"]) for item in batch)
    frames_padded: List[torch.Tensor] = []
    frame_masks: List[torch.Tensor] = []
    labels_padded: List[List[Optional[Dict[str, Any]]]] = []
    target_padded: List[List[Optional[torch.Tensor]]] = []
    indices_padded: List[List[int]] = []

    for item in batch:
        clip = item["frames"]
        t = int(item["length"])
        pad_t = max_t - t
        if pad_t > 0:
            pad_frame = clip[-1:].repeat(pad_t, 1, 1, 1)
            clip = torch.cat([clip, pad_frame], dim=0)

        frames_padded.append(clip)

        mask = torch.zeros(max_t, dtype=torch.bool)
        mask[:t] = True
        frame_masks.append(mask)

        labels_seq = list(item.get("labels", []))
        labels_seq = labels_seq + [None] * (max_t - len(labels_seq))
        labels_padded.append(labels_seq)

        target_seq = list(item.get("target_features", []))
        target_seq = target_seq + [None] * (max_t - len(target_seq))
        target_padded.append(target_seq)

        idx_seq = list(item.get("frame_indices", []))
        fill_idx = idx_seq[-1] if len(idx_seq) > 0 else -1
        idx_seq = idx_seq + [fill_idx] * (max_t - len(idx_seq))
        indices_padded.append(idx_seq)

    frames = torch.stack(frames_padded, dim=0)
    frame_mask = torch.stack(frame_masks, dim=0)
    labels = labels_padded
    target_features = target_padded
    video_ids = [item["video_id"] for item in batch]
    return {
        "frames": frames,
        "frame_mask": frame_mask,
        "frame_indices": indices_padded,
        "lengths": [int(item["length"]) for item in batch],
        "labels": labels,
        "target_features": target_features,
        "video_ids": video_ids,
    }
