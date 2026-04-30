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

class VSPWDataset(Dataset):
    """
    Simplified VSPW dataset with augmentation (single-frame).
    """

    def __init__(
        self,
        root: Path,
        split_dir: str = "train",
        num_classes: int = 124,
        image_size: Tuple[int, int] = (384, 640),
        max_samples: Optional[int] = None,
        seed: int = 42,
        augment: bool = True,
        seq_len: int = 8,
    ):
        super().__init__()

        self.root = Path(root)
        self.data_root = self.root / "data"
        self.split_file = self.root / f"{split_dir}.txt"
        self.split_dir = split_dir
        self.num_classes = num_classes
        self.image_size = image_size
        self.augment = augment

        self.rng = random.Random(seed)

        self.items = self._collect_items(max_samples, seed)
        if not self.items:
            raise RuntimeError("No samples found.")

        self.to_tensor = transforms.ToTensor()

        self.resize_img = transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.resize_mask = transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.NEAREST
        )

        # color jitter (ONLY image)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )
        self.seq_len = max(1, int(seq_len))

    # -------------------------
    # split file
    # -------------------------
    def _load_video_list(self) -> Set[str]:
        if not self.split_file.exists():
            return set()

        with self.split_file.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    # -------------------------
    # collect frames
    # -------------------------
    def _collect_items(self, max_samples, seed) -> List[VSPWItem]:
        video_filter = self._load_video_list()
        items: List[VSPWItem] = []

        for video_dir in sorted(self.data_root.iterdir()):
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name
            if video_filter and video_id not in video_filter:
                continue

            img_dir = video_dir / "origin"
            mask_dir = video_dir / "mask"

            if not img_dir.exists() or not mask_dir.exists():
                continue

            for idx, mask_path in enumerate(sorted(mask_dir.glob("*.png"))):
                
                if mask_path.name.startswith("._") or idx > self.seq_len - 1:
                    continue

                frame_id = mask_path.stem
                img_path = img_dir / f"{frame_id}.jpg"

                if not img_path.exists():
                    continue

                items.append(
                    VSPWItem(img_path, mask_path, video_id, frame_id)
                )
        return items

    # -------------------------
    # augmentation (IMPORTANT)
    # -------------------------
    def _augment(self, image, mask):
        # Random horizontal flip
        if self.rng.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        if self.rng.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if self.rng.random() < 0.5:
            angle = self.rng.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        return image, mask

    def _crop(self, image, mask):
        if self.split_dir == "train":
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.image_size
            )
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        else:
            # center crop
            image = TF.center_crop(image, self.image_size)
            mask = TF.center_crop(mask, self.image_size)
        return image, mask
    
    # -------------------------
    # main loader
    # -------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]

        image = Image.open(item.image_path).convert("RGB")
        mask = Image.open(item.mask_path)

        # resize first (stable baseline)
        image, mask = self._crop(image, mask)

        # augmentation
        if self.augment:
            image, mask = self._augment(image, mask)

        image = self.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # invalid labels
        mask[mask >= self.num_classes] = 255

        return {
            "image": image,
            "mask": mask,
            "video_id": item.video_id,
            "frame_id": item.frame_id,
        }

    def __len__(self):
        return len(self.items)
    

class VSPWSequenceDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split_dir: str = "train",
        num_classes: int = 124,
        image_size: Tuple[int, int] = (384, 640),
        max_samples: Optional[int] = None,
        seed: int = 42,
        augment: bool = True,
        seq_len: int = 8,
    ):
        super().__init__()

        self.root = Path(root)
        self.data_root = self.root / "data"
        self.split_file = self.root / f"{split_dir}.txt"

        self.num_classes = num_classes
        self.image_size = image_size
        self.augment = augment

        self.rng = random.Random(seed)

        self.items = self._collect_items(max_samples, seed)
        if not self.items:
            raise RuntimeError("No samples found.")

        self.to_tensor = transforms.ToTensor()

        self.resize_img = transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.resize_mask = transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.NEAREST
        )

        self.seq_len = max(1, int(seq_len))

    # -------------------------
    # split file
    # -------------------------
    def _load_video_list(self) -> Set[str]:
        if not self.split_file.exists():
            return set()

        with self.split_file.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    # -------------------------
    # collect frames
    # -------------------------
    def _collect_items(self, max_samples, seed) -> List[VSPWItem]:
        video_filter = self._load_video_list()
        items: List[List[VSPWItem]] = []
        seq_items: List[VSPWItem] = []
        for video_dir in sorted(self.data_root.iterdir()):
            seq_items = []
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name
            if video_filter and video_id not in video_filter:
                continue

            img_dir = video_dir / "origin"
            mask_dir = video_dir / "mask"

            if not img_dir.exists() or not mask_dir.exists():
                continue

            for idx, mask_path in enumerate(sorted(mask_dir.glob("*.png"))):
                
                if mask_path.name.startswith("._") or idx > self.seq_len - 1:
                    continue

                frame_id = mask_path.stem
                img_path = img_dir / f"{frame_id}.jpg"

                if not img_path.exists():
                    continue

                seq_items.append(
                    VSPWItem(img_path, mask_path, video_id, frame_id)
                )
            items.append(seq_items)
        return items

    # -------------------------
    # augmentation (IMPORTANT)
    # -------------------------
    def _augment_sequence(self, images, masks):
        # random flip decision (shared)
        if self.rng.random() < 0.5:
            images = [TF.hflip(img) for img in images]
            masks = [TF.hflip(m) for m in masks]
        if self.rng.random() < 0.5:
            images = [TF.vflip(img) for img in images]
            masks = [TF.vflip(m) for m in masks]
        if self.rng.random() < 0.5:
            angle = self.rng.uniform(-10, 10)
            images = [TF.rotate(img, angle) for img in images]
            masks = [TF.rotate(m, angle) for m in masks]
        return images, masks

    def _crop_sequence(self, images, masks):
        if self.split_dir == "train":
            i, j, h, w = transforms.RandomCrop.get_params(
                images[0], output_size=self.image_size
            )
            images = [TF.crop(img, i, j, h, w) for img in images]
            masks = [TF.crop(m, i, j, h, w) for m in masks]
        else:
            images = [TF.center_crop(img, self.image_size) for img in images]
            masks = [TF.center_crop(m, self.image_size) for m in masks]

        return images, masks
    # -------------------------
    # main loader
    # -------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_items = self.items[idx]
        images = []
        masks = []
        video_id = seq_items[0].video_id
        frame_ids = []
        for idx, item in enumerate(seq_items):
            image = Image.open(item.image_path).convert("RGB")
            mask = Image.open(item.mask_path)

            # resize first (stable baseline)
            images.append(image)
            masks.append(mask)
            frame_ids.append(item.frame_id)
        
        images, masks = self._crop_sequence(images, masks)

        # ---- augment (sync) ----
        if self.augment:
            images, masks = self._augment_sequence(images, masks)

        # ---- to tensor ----
        images = [self.to_tensor(img) for img in images]
        masks = [torch.as_tensor(np.array(m), dtype=torch.long) for m in masks]

        # ---- stack ----
        images = torch.stack(images, dim=0)   # [T, C, H, W]
        masks = torch.stack(masks, dim=0)     # [T, H, W]

        # ---- invalid label ----
        masks[masks >= self.num_classes] = 255

        return {
            "image": image,
            "mask": mask,
            "video_id": item.video_id,
            "frame_id": item.frame_id,
        }

    def __len__(self):
        return len(self.items)