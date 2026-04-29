from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass
class VSPWItem:
    image_path: Path
    mask_path: Path
    video_id: str
    frame_id: str


class SemanticSegmentationDataset(Dataset):
    """Semantic segmentation dataset for VSPW only.

    Expected VSPW layout:
      root/
        data/<video_id>/origin/<frame>.jpg
        data/<video_id>/mask/<frame>.png
        train.txt | val.txt | test.txt
    """

    def __init__(
        self,
        split_dir: Path,
        num_classes: int = 124,
        vspw_split: Optional[str] = None,
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
        self.num_classes = int(num_classes)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.spatial_transform = str(spatial_transform).lower()
        self.crop_type = str(crop_type).lower()
        self.pad_if_needed = bool(pad_if_needed)

        if self.spatial_transform not in ("resize", "crop"):
            raise ValueError("spatial_transform must be one of {'resize', 'crop'}")
        if self.crop_type not in ("random", "center"):
            raise ValueError("crop_type must be one of {'random', 'center'}")

        self.root = self._resolve_vspw_root(Path(split_dir))
        self.data_root = self.root / "data"
        if not self.data_root.exists() or not self.data_root.is_dir():
            raise FileNotFoundError(f"Missing VSPW data directory: {self.data_root}")

        self.vspw_split = self._normalize_split_name(vspw_split or self._infer_split_from_path(Path(split_dir)))

        self.seq_len = max(1, int(seq_len))
        self.seq_stride = self.seq_len if seq_stride is None else max(1, int(seq_stride))

        self.items = self._collect_items(max_samples=max_samples, seed=seed)
        if not self.items:
            raise RuntimeError(f"No VSPW image/mask pairs found under {self.root}")

        self._build_sequence_index()

        h, w = self.image_size
        self.to_tensor = transforms.ToTensor()
        self.norm_image_tf = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.image_resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)

    def _resolve_vspw_root(self, split_dir: Path) -> Path:
        # Accept either VSPW root itself or a child path like root/train, root/val, root/test.
        if (split_dir / "data").exists():
            return split_dir
        if (split_dir.parent / "data").exists():
            return split_dir.parent
        raise FileNotFoundError(
            f"Cannot locate VSPW root from split_dir={split_dir}. Expected '<root>/data' directory."
        )

    def _infer_split_from_path(self, split_dir: Path) -> Optional[str]:
        name = split_dir.name.lower()
        if name in {"train", "tr"}:
            return "train"
        if name in {"val", "valid", "validation"}:
            return "val"
        if name in {"test", "te"}:
            return "test"
        return None

    def _normalize_split_name(self, split: Optional[str]) -> Optional[str]:
        if split is None:
            return None
        s = str(split).strip().lower()
        aliases = {"valid": "val", "validation": "val"}
        return aliases.get(s, s)

    def _load_split_video_filter(self) -> Optional[Set[str]]:
        if self.vspw_split is None:
            return None

        split_file = self.root / f"{self.vspw_split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"VSPW split file not found: {split_file}. "
                "Available splits are typically train.txt, val.txt, test.txt."
            )

        with split_file.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _collect_items(self, max_samples: Optional[int], seed: int) -> List[VSPWItem]:
        videos_filter = self._load_split_video_filter()
        pairs: List[VSPWItem] = []

        for video_dir in sorted(self.data_root.iterdir()):
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name
            if videos_filter is not None and video_id not in videos_filter:
                continue

            img_dir = video_dir / "origin"
            mask_dir = video_dir / "mask"
            if not img_dir.exists() or not mask_dir.exists():
                continue

            for mask_path in sorted(mask_dir.glob("*.png")):
                if mask_path.name.startswith("._"):
                    continue

                frame_id = mask_path.stem
                img_path = img_dir / f"{frame_id}.jpg"
                if not img_path.exists():
                    continue

                pairs.append(
                    VSPWItem(
                        image_path=img_path,
                        mask_path=mask_path,
                        video_id=video_id,
                        frame_id=frame_id,
                    )
                )

        if max_samples is not None and max_samples < len(pairs):
            rng = random.Random(seed)
            rng.shuffle(pairs)
            pairs = pairs[: max(1, int(max_samples))]

        return pairs

    def _build_sequence_index(self) -> None:
        videos: Dict[str, List[VSPWItem]] = {}
        for item in self.items:
            videos.setdefault(item.video_id, []).append(item)

        for video_id in videos:
            videos[video_id] = sorted(videos[video_id], key=lambda x: x.frame_id)

        self.videos = videos
        self.sequence_index: List[Tuple[str, int]] = []

        if self.seq_len <= 1:
            return

        for video_id, frames in self.videos.items():
            n = len(frames)
            if n < self.seq_len:
                continue
            for start in range(0, n - self.seq_len + 1, self.seq_stride):
                self.sequence_index.append((video_id, start))

    def __len__(self) -> int:
        if self.seq_len <= 1:
            return len(self.items)
        return len(self.sequence_index) if self.sequence_index else len(self.items)

    def _crop_pair(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        target_h, target_w = self.image_size

        if self.pad_if_needed:
            pad_h = max(0, target_h - image.height)
            pad_w = max(0, target_w - image.width)
            if pad_h > 0 or pad_w > 0:
                pad = [0, 0, pad_w, pad_h]
                image = TF.pad(image, pad, fill=0)
                mask = TF.pad(mask, pad, fill=0)

        if self.crop_type == "random":
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(target_h, target_w))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            return image, mask

        image = TF.center_crop(image, [target_h, target_w])
        mask = TF.center_crop(mask, [target_h, target_w])
        return image, mask

    def _prepare_item(self, item: VSPWItem) -> Dict[str, torch.Tensor]:
        image = Image.open(item.image_path).convert("RGB")
        # Keep original indexed labels (e.g., palette PNG mode "P").
        # Converting to "L" changes class ids into luminance values.
        mask = Image.open(item.mask_path)

        if self.spatial_transform == "crop":
            image, mask = self._crop_pair(image, mask)
        else:
            image = self.image_resize(image)
            mask = self.mask_resize(mask)

        image_raw = self.to_tensor(image)
        image_norm = self.norm_image_tf(image_raw.clone())

        mask_np = np.array(mask, dtype=np.int64)
        mask_np[(mask_np >= self.num_classes)] = 255
        target = torch.from_numpy(mask_np)

        return {
            "image": image_norm,
            "image_raw": image_raw,
            "mask": target,
            "video_id": item.video_id,
            "frame_id": item.frame_id,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.seq_len > 1 and self.sequence_index:
            video_id, start = self.sequence_index[index]
            frames = self.videos[video_id][start : start + self.seq_len]

            outputs = [self._prepare_item(item) for item in frames]
            return {
                "image": torch.stack([o["image"] for o in outputs], dim=0),
                "image_raw": torch.stack([o["image_raw"] for o in outputs], dim=0),
                "mask": torch.stack([o["mask"] for o in outputs], dim=0),
                "video_id": video_id,
                "frame_id": [o["frame_id"] for o in outputs],
            }

        return self._prepare_item(self.items[index])
