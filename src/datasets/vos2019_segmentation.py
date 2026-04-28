from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass
class VOSItem:
	image_path: Path
	mask_path: Path


class VOS2019BinarySegmentationDataset(Dataset):
	"""Binary segmentation dataset for VOS2019/YTVOS-style folder layout.

	Expected split structure:
	  split_dir/
	    JPEGImages/<video_id>/<frame>.jpg
	    Annotations/<video_id>/<frame>.png

	Mask values are collapsed into binary classes:
	  background=0, foreground=1 (all object IDs > 0).
	"""

	def __init__(
		self,
		split_dir: Path,
		image_size: Tuple[int, int] = (384, 640),
		spatial_transform: str = "resize",
		crop_type: str = "center",
		pad_if_needed: bool = True,
		max_samples: Optional[int] = None,
		seed: int = 42,
		seq_len: int = 1,
		seq_stride: Optional[int] = None,
	) -> None:
		super().__init__()
		self.split_dir = Path(split_dir)
		self.img_root = self.split_dir / "JPEGImages"
		self.mask_root = self.split_dir / "Annotations"

		if not self.img_root.exists() or not self.mask_root.exists():
			raise FileNotFoundError(
				f"Missing JPEGImages/Annotations under {self.split_dir}."
			)

		# Support returning short sequences of consecutive frames per item.
		self.seq_len = max(1, int(seq_len))
		# stride between sequences (default to seq_len if not provided)
		if seq_stride is None:
			self.seq_stride = self.seq_len
		else:
			self.seq_stride = max(1, int(seq_stride))
		# items is kept for backward-compatibility when seq_len == 1
		self.items = self._collect_items(max_samples=max_samples, seed=seed)
		# build per-video frame lists and sequence index for seq_len > 1
		self._build_sequence_index()

		if len(self.items) == 0:
			raise RuntimeError(f"No image/mask pairs found under {self.split_dir}")

		h, w = int(image_size[0]), int(image_size[1])
		self.image_size = (h, w)
		self.spatial_transform = str(spatial_transform).lower()
		self.crop_type = str(crop_type).lower()
		self.pad_if_needed = bool(pad_if_needed)

		if self.spatial_transform not in ("resize", "crop"):
			raise ValueError("spatial_transform must be one of {'resize', 'crop'}")
		if self.crop_type not in ("random", "center"):
			raise ValueError("crop_type must be one of {'random', 'center'}")

		self.to_tensor = transforms.ToTensor()
		self.norm_image_tf = transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225],
		)
		self.image_resize = transforms.Resize(
			(h, w), interpolation=transforms.InterpolationMode.BILINEAR
		)
		self.mask_resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)

	def _collect_items(self, max_samples: Optional[int], seed: int) -> List[VOSItem]:
		pairs: List[VOSItem] = []
		for video_dir in sorted(self.mask_root.iterdir()):
			if not video_dir.is_dir():
				continue
			for mask_path in sorted(video_dir.glob("*.png")):
				img_path = self.img_root / video_dir.name / f"{mask_path.stem}.jpg"
				if img_path.exists():
					pairs.append(VOSItem(image_path=img_path, mask_path=mask_path))

		if max_samples is not None and max_samples < len(pairs):
			rng = random.Random(seed)
			rng.shuffle(pairs)
			pairs = pairs[: max(1, int(max_samples))]

		return pairs

	def _build_sequence_index(self) -> None:
		"""Build helper structures to support sequence sampling.

		Creates:
		  - self.videos: dict video_id -> ordered list[VOSItem]
		  - self.sequence_index: list of (video_id, start_idx) for valid sequences
		"""
		from collections import defaultdict

		videos = defaultdict(list)
		for it in self.items:
			video_id = it.image_path.parent.name
			videos[video_id].append(it)

		# ensure each video's frames are sorted by filename (already ensured in collect)
		self.videos = {k: v for k, v in videos.items()}
		self.sequence_index = []
		if self.seq_len <= 1:
			# no special sequence indexing required
			return

		for vid, frames in self.videos.items():
			n = len(frames)
			if n < self.seq_len:
				continue
			for start in range(0, n - self.seq_len + 1, self.seq_stride):
				self.sequence_index.append((vid, start))

		# if no sequences found, fall back to flat items
		if len(self.sequence_index) == 0:
			self.sequence_index = []

	def __len__(self) -> int:
		if self.seq_len <= 1:
			return len(self.items)
		# number of valid sequences
		return len(self.sequence_index) if hasattr(self, "sequence_index") and len(self.sequence_index) > 0 else len(self.items)

	def _crop_pair(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
		target_h, target_w = self.image_size

		if self.pad_if_needed:
			pad_h = max(0, target_h - image.height)
			pad_w = max(0, target_w - image.width)
			if pad_h > 0 or pad_w > 0:
				# left, top, right, bottom
				pad = [0, 0, pad_w, pad_h]
				image = TF.pad(image, pad, fill=0)
				mask = TF.pad(mask, pad, fill=0)

		if self.crop_type == "random":
			i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(target_h, target_w))
			image = TF.crop(image, i, j, h, w)
			mask = TF.crop(mask, i, j, h, w)
			return image, mask

		# center crop
		image = TF.center_crop(image, [target_h, target_w])
		mask = TF.center_crop(mask, [target_h, target_w])
		return image, mask

	def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
		# If seq_len > 1 and sequence_index exists, return a T-length sequence
		if self.seq_len > 1 and hasattr(self, "sequence_index") and len(self.sequence_index) > 0:
			vid, start = self.sequence_index[index]
			frames = self.videos[vid][start : start + self.seq_len]
			images = []
			images_raw = []
			masks = []
			frame_ids = []
			for it in frames:
				image = Image.open(it.image_path).convert("RGB")
				mask = Image.open(it.mask_path).convert("L")

				if self.spatial_transform == "crop":
					image, mask = self._crop_pair(image, mask)
				else:
					image = self.image_resize(image)
					mask = self.mask_resize(mask)

				image_raw = self.to_tensor(image)
				image_norm = self.norm_image_tf(image_raw.clone())

				mask_np = np.array(mask, dtype=np.uint8)
				mask_bin = (mask_np > 0).astype(np.int64)
				target = torch.from_numpy(mask_bin)

				images.append(image_norm)
				images_raw.append(image_raw)
				masks.append(target)
				frame_ids.append(it.image_path.stem)

			# Stack along time dimension T as first dim
			image_seq = torch.stack(images, dim=0)
			image_raw_seq = torch.stack(images_raw, dim=0)
			mask_seq = torch.stack(masks, dim=0)

			return {
				"image": image_seq,
				"image_raw": image_raw_seq,
				"mask": mask_seq,
				"video_id": vid,
				"frame_id": frame_ids,
			}

		# Fallback: seq_len == 1 behavior (existing)
		item = self.items[index]
		image = Image.open(item.image_path).convert("RGB")
		mask = Image.open(item.mask_path).convert("L")

		if self.spatial_transform == "crop":
			image, mask = self._crop_pair(image, mask)
		else:
			image = self.image_resize(image)
			mask = self.mask_resize(mask)

		image_raw = self.to_tensor(image)
		image_norm = self.norm_image_tf(image_raw.clone())

		mask_np = np.array(mask, dtype=np.uint8)
		mask_bin = (mask_np > 0).astype(np.int64)
		target = torch.from_numpy(mask_bin)

		return {
			"image": image_norm,
			"image_raw": image_raw,
			"mask": target,
			"video_id": item.image_path.parent.name,
			"frame_id": item.image_path.stem,
		}

