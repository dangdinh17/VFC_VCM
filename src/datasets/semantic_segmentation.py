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
        vspw_split: Optional[str] = 'train',
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

        self.root = Path(split_dir)
        self.data_root = self.root / "data"
        self.split_file = self.root / f"{vspw_split}.txt"
        
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


    def _load_split_video_filter(self) -> Optional[Set[str]]:
        with self.split_file.open("r", encoding="utf-8") as f:
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

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask[(mask >= self.num_classes)] = 255

        return {
            "image": image_norm,
            "image_raw": image_raw,
            "mask": mask,
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

import lmdb
import io
from PIL import Image
import torch
import numpy as np

class LMDBBase:
    def __init__(self, lmdb_path: str):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=True,
            readahead=False,
            meminit=False,
        )

    def _get(self, key: str) -> bytes:
        with self.env.begin(write=False) as txn:
            val = txn.get(key.encode())
        return val

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class LMDBFrameDataset(LMDBBase, Dataset):
    def __init__(self, lmdb_path, index_list, num_classes=124, image_size=(384, 640)):
        LMDBBase.__init__(self, lmdb_path)
        self.index = index_list  # list of (vid, frame)
        self.num_classes = num_classes
        self.size = image_size

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vid, frame = self.index[i]

        img_bytes = self._get(f"{vid}/{frame}_img")
        mask_bytes = self._get(f"{vid}/{frame}_mask")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes))

        img = TF.resize(img, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask[mask >= self.num_classes] = 255

        return {
            "image": img,
            "mask": mask,
            "frame_id": frame,
            "video_id": vid,
        }
        
class LMDBSequenceDataset(LMDBBase, Dataset):
    def __init__(
        self,
        lmdb_path,
        index_list,   # sorted list per video
        video_map,    # {vid: [frame1, frame2,...]}
        seq_len=5,
        num_classes=124,
        image_size=(384, 640),
    ):
        LMDBBase.__init__(self, lmdb_path)

        self.video_map = video_map
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.size = image_size

        self.samples = []

        for vid, frames in video_map.items():
            for i in range(len(frames) - seq_len + 1):
                self.samples.append((vid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vid, start = self.samples[i]
        frames = self.video_map[vid][start:start + self.seq_len]

        imgs = []
        masks = []

        for frame in frames:
            img_bytes = self._get(f"{vid}/{frame}_img")
            mask_bytes = self._get(f"{vid}/{frame}_mask")

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            mask = Image.open(io.BytesIO(mask_bytes))

            img = TF.resize(img, self.size)
            mask = TF.resize(mask, self.size, TF.InterpolationMode.NEAREST)

            imgs.append(TF.to_tensor(img))
            masks.append(torch.as_tensor(np.array(mask), dtype=torch.long))

        imgs = torch.stack(imgs)
        masks = torch.stack(masks)

        masks[masks >= self.num_classes] = 255

        return {
            "image": imgs,   # [T, C, H, W]
            "mask": masks,   # [T, H, W]
            "video_id": vid,
            "frame_id": frames,
        }