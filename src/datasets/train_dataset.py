"""Training dataset for ROI-VFC video feature codec pipeline.

Temporal-aware dataset output:
- returns frame sequence `[T, C, H, W]`
- returns aligned labels per timestep
- supports variable-length clips
- provides mask/padding metadata in collate
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SampleLike = Dict[str, Any]


def _load_manifest(manifest: Union[str, Path, Sequence[SampleLike]]) -> List[SampleLike]:
	if isinstance(manifest, (str, Path)):
		with open(manifest, "r", encoding="utf-8") as f:
			data = json.load(f)
		if isinstance(data, dict) and "samples" in data:
			return data["samples"]
		if isinstance(data, list):
			return data
		raise ValueError("Manifest JSON must be a list or a dict containing 'samples'.")
	return list(manifest)


def _ensure_float_image(img: torch.Tensor) -> torch.Tensor:
	if img.dtype != torch.float32:
		img = img.float()
	if img.max() > 1.5:
		img = img / 255.0
	return img


def _read_image(path: Union[str, Path], image_size: Optional[Sequence[int]]) -> torch.Tensor:
	img = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(f"Cannot read image: {path}")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
	if image_size is not None:
		h, w = int(image_size[0]), int(image_size[1])
		t = F.interpolate(t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
	return _ensure_float_image(t)


def _read_video_frames(
	video_path: Union[str, Path],
	indices: Sequence[int],
	image_size: Optional[Sequence[int]],
) -> torch.Tensor:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise FileNotFoundError(f"Cannot open video: {video_path}")

	frames: List[torch.Tensor] = []
	for idx in indices:
		cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
		ok, frame = cap.read()
		if not ok:
			raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		ft = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float()
		if image_size is not None:
			h, w = int(image_size[0]), int(image_size[1])
			ft = F.interpolate(ft.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
		frames.append(_ensure_float_image(ft))

	cap.release()
	return torch.stack(frames, dim=0)


def _load_optional_tensor(path_like: Optional[Union[str, Path]]) -> Optional[torch.Tensor]:
	if path_like is None:
		return None
	p = Path(path_like)
	if not p.exists():
		return None
	return torch.load(str(p), map_location="cpu")


def _as_long_mask(mask: torch.Tensor, image_size: Optional[Sequence[int]]) -> torch.Tensor:
	if mask.dtype != torch.long:
		mask = mask.long()
	if image_size is not None:
		h, w = int(image_size[0]), int(image_size[1])
		mask = F.interpolate(mask[None, None].float(), size=(h, w), mode="nearest")[0, 0].long()
	return mask


def _load_segmentation_mask(path_like: Optional[Union[str, Path]], image_size: Optional[Sequence[int]]) -> Optional[torch.Tensor]:
	if path_like is None:
		return None
	mask = cv2.imread(str(path_like), cv2.IMREAD_GRAYSCALE)
	if mask is None:
		return None
	t = torch.from_numpy(mask)
	return _as_long_mask(t, image_size)


def _safe_take(seq: Any, index: int) -> Any:
	if isinstance(seq, (list, tuple)) and len(seq) > 0:
		return seq[min(index, len(seq) - 1)]
	return None


def _resolve_total_frames(sample: SampleLike) -> int:
	if "frames" in sample:
		return len(sample["frames"])
	if "num_frames" in sample:
		return max(1, int(sample["num_frames"]))
	if "video" in sample:
		cap = cv2.VideoCapture(str(sample["video"]))
		if cap.isOpened():
			total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			cap.release()
			return max(1, total)
		cap.release()
	return 1


def _align_labels_for_indices(
	sample: SampleLike,
	indices: Sequence[int],
	image_size: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
	"""Align machine-task labels to selected frame indices.

	Supported conventions:
	- `sample["frame_labels"]`: list[dict] aligned with original timeline.
	- `sample["labels"][task]` as list -> take by frame index.
	- `sample["labels"]["segmentation_mask_paths"]` as list -> load per frame.
	- `sample["labels"]["segmentation_mask_path"]` as scalar -> shared for all frames.
	"""
	base_labels = sample.get("labels", {})
	if not isinstance(base_labels, dict):
		base_labels = {}

	per_frame_labels = sample.get("frame_labels")

	aligned: List[Dict[str, Any]] = []
	for idx in indices:
		cur: Dict[str, Any] = {}

		# 1) Global/static labels
		for key, val in base_labels.items():
			if isinstance(val, (list, tuple)):
				cur[key] = _safe_take(val, idx)
			else:
				cur[key] = val

		# 2) Optional per-frame label dict
		if isinstance(per_frame_labels, (list, tuple)) and len(per_frame_labels) > 0:
			frame_label = _safe_take(per_frame_labels, idx)
			if isinstance(frame_label, dict):
				cur.update(frame_label)

		# 3) Convert segmentation paths to actual tensor mask
		seg_paths = cur.get("segmentation_mask_paths")
		if isinstance(seg_paths, (list, tuple)):
			seg_path = _safe_take(seg_paths, idx)
			seg = _load_segmentation_mask(seg_path, image_size)
			if seg is not None:
				cur["segmentation_mask"] = seg
			cur.pop("segmentation_mask_paths", None)

		if "segmentation_mask" not in cur:
			seg = _load_segmentation_mask(cur.get("segmentation_mask_path"), image_size)
			if seg is not None:
				cur["segmentation_mask"] = seg

		aligned.append(cur)

	return aligned


def _align_target_features_for_indices(
	sample: SampleLike,
	indices: Sequence[int],
) -> List[Optional[torch.Tensor]]:
	"""Align optional target features by timestep."""
	seq_paths = sample.get("target_features_paths")
	if isinstance(seq_paths, (list, tuple)) and len(seq_paths) > 0:
		out: List[Optional[torch.Tensor]] = []
		for idx in indices:
			p = _safe_take(seq_paths, idx)
			out.append(_load_optional_tensor(p))
		return out

	tf = _load_optional_tensor(sample.get("target_features_path"))
	if torch.is_tensor(tf) and tf.dim() >= 1 and tf.size(0) >= 1:
		if tf.size(0) >= max(indices) + 1:
			return [tf[idx] for idx in indices]
		return [tf[min(i, tf.size(0) - 1)] for i in indices]

	return [tf for _ in indices]


class TrainVideoFeatureDataset(Dataset):
	"""Dataset returning clips + labels for ROI-VFC downstream training.

	Expected per-sample keys:
	  - one of: `frames` (list of paths) OR `video` (video file path)
	  - optional `labels`: dict for downstream tasks
	  - optional `target_features_path`: .pt file path
	  - optional `video_id`
	"""

	def __init__(
		self,
		manifest: Union[str, Path, Sequence[SampleLike]],
		clip_length: int = 2,
		image_size: Optional[Sequence[int]] = (256, 256),
		random_flip: bool = True,
	) -> None:
		super().__init__()
		self.samples = _load_manifest(manifest)
		self.clip_length = int(clip_length)
		self.image_size = image_size
		self.random_flip = random_flip

		if self.clip_length < 1:
			raise ValueError("clip_length must be >= 1")

	def __len__(self) -> int:
		return len(self.samples)

	def _sample_indices(self, num_frames: int) -> List[int]:
		effective = min(num_frames, self.clip_length)
		if num_frames <= self.clip_length:
			return list(range(effective))
		start = random.randint(0, num_frames - self.clip_length)
		return list(range(start, start + self.clip_length))

	def _read_clip(self, sample: SampleLike) -> Tuple[torch.Tensor, List[int]]:
		if "frames" in sample:
			frame_paths = sample["frames"]
			if len(frame_paths) == 0:
				raise ValueError("Sample has empty 'frames' list")
			indices = self._sample_indices(len(frame_paths))
			frames = [_read_image(frame_paths[i], self.image_size) for i in indices]
			return torch.stack(frames, dim=0), indices

		if "video" in sample:
			num_frames = _resolve_total_frames(sample)
			indices = self._sample_indices(max(num_frames, 1))
			return _read_video_frames(sample["video"], indices, self.image_size), indices

		raise ValueError("Each sample must contain either 'frames' or 'video'.")

	def __getitem__(self, index: int) -> Dict[str, Any]:
		sample = self.samples[index]
		clip, frame_indices = self._read_clip(sample)
		length = int(clip.size(0))

		if self.random_flip and random.random() < 0.5:
			clip = torch.flip(clip, dims=[3])

		labels = _align_labels_for_indices(sample, frame_indices, self.image_size)
		target_features = _align_target_features_for_indices(sample, frame_indices)

		return {
			"frames": clip,  # [T, C, H, W]
			"labels": labels,
			"target_features": target_features,
			"frame_indices": frame_indices,
			"length": length,
			"video_id": sample.get("video_id", f"sample_{index}"),
		}


def train_video_feature_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
	# Pad clips to max temporal length and provide frame validity mask.
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

	frames = torch.stack(frames_padded, dim=0)  # [B, T, C, H, W]
	frame_mask = torch.stack(frame_masks, dim=0)  # [B, T]
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
