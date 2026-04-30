from __future__ import annotations

import io
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
import lmdb


@dataclass(frozen=True)
class VSPWLMDBItem:
    video_id: str
    frame_id: str
    image_key: bytes
    mask_key: bytes


class _VSPWLMDBBase(Dataset):
    """LMDB-backed VSPW semantic segmentation dataset.

    LMDB keys are expected in the form built by ``src/utils/lmdb.py``:
    - ``{video_id}/{frame_id}_img``
    - ``{video_id}/{frame_id}_mask``
    """

    def __init__(
        self,
        split_dir: Path,
        num_classes: int = 124,
        vspw_split: Optional[str] = "train",
        image_size: Tuple[int, int] = (384, 640),
        spatial_transform: str = "resize",
        crop_type: str = "center",
        pad_if_needed: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
        seq_len: int = 1,
        seq_stride: Optional[int] = None,
        lmdb_path: Optional[Path] = None,
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
        self.split_file = self.root / f"{vspw_split}.txt"
        self.lmdb_path = Path(lmdb_path) if lmdb_path is not None else (self.root / "vspw.lmdb")

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path not found: {self.lmdb_path}")

        self.seq_len = max(1, int(seq_len))
        self.seq_stride = self.seq_len if seq_stride is None else max(1, int(seq_stride))

        self._env: Optional[lmdb.Environment] = None
        self._txn: Optional[lmdb.Transaction] = None

        self.items = self._collect_items(max_samples=max_samples, seed=seed)
        if not self.items:
            raise RuntimeError(f"No VSPW LMDB image/mask pairs found in {self.lmdb_path}")

        self._build_sequence_index()

        h, w = self.image_size
        self.to_tensor = transforms.ToTensor()
        self.norm_image_tf = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.image_resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._txn is not None:
            self._txn.abort()
            self._txn = None
        if self._env is not None:
            self._env.close()
            self._env = None

    def _open_env(self) -> lmdb.Environment:
        return lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2048,
        )

    def _ensure_txn(self) -> lmdb.Transaction:
        if self._env is None:
            self._env = self._open_env()
            self._txn = self._env.begin(write=False)
        if self._txn is None:
            self._txn = self._env.begin(write=False)
        return self._txn

    def _load_split_video_filter(self) -> Optional[Set[str]]:
        with self.split_file.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    @staticmethod
    def _parse_key(key: bytes) -> Optional[Tuple[str, str, str]]:
        try:
            key_str = key.decode("utf-8")
        except UnicodeDecodeError:
            return None

        if "/" not in key_str:
            return None
        video_id, frame_part = key_str.split("/", 1)

        if frame_part.endswith("_img"):
            frame_id = frame_part[: -len("_img")]
            kind = "img"
        elif frame_part.endswith("_mask"):
            frame_id = frame_part[: -len("_mask")]
            kind = "mask"
        else:
            return None

        if not video_id or not frame_id:
            return None
        return video_id, frame_id, kind

    def _collect_items(self, max_samples: Optional[int], seed: int) -> List[VSPWLMDBItem]:
        videos_filter = self._load_split_video_filter()
        pair_flags: Dict[Tuple[str, str], int] = {}

        env = self._open_env()
        try:
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    parsed = self._parse_key(key)
                    if parsed is None:
                        continue

                    video_id, frame_id, kind = parsed
                    if videos_filter is not None and video_id not in videos_filter:
                        continue

                    k = (video_id, frame_id)
                    flags = pair_flags.get(k, 0)
                    if kind == "img":
                        flags |= 1
                    else:
                        flags |= 2
                    pair_flags[k] = flags
        finally:
            env.close()

        pairs: List[VSPWLMDBItem] = []
        for (video_id, frame_id), flags in sorted(pair_flags.items()):
            if flags != 3:
                continue
            pairs.append(
                VSPWLMDBItem(
                    video_id=video_id,
                    frame_id=frame_id,
                    image_key=f"{video_id}/{frame_id}_img".encode("utf-8"),
                    mask_key=f"{video_id}/{frame_id}_mask".encode("utf-8"),
                )
            )

        if max_samples is not None and max_samples < len(pairs):
            rng = random.Random(seed)
            rng.shuffle(pairs)
            pairs = pairs[: max(1, int(max_samples))]

        return pairs

    def _build_sequence_index(self) -> None:
        videos: Dict[str, List[VSPWLMDBItem]] = {}
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

    def _fetch_bytes(self, key: bytes) -> bytes:
        txn = self._ensure_txn()
        value = txn.get(key)
        if value is None:
            raise KeyError(f"LMDB key not found: {key.decode('utf-8', errors='ignore')}")
        return bytes(value)

    def _prepare_item(self, item: VSPWLMDBItem) -> Dict[str, torch.Tensor]:
        img_bytes = self._fetch_bytes(item.image_key)
        mask_bytes = self._fetch_bytes(item.mask_key)

        with Image.open(io.BytesIO(img_bytes)) as im:
            image = im.convert("RGB")

        # Keep original indexed labels (e.g., palette PNG mode "P").
        # Converting to "L" changes class ids into luminance values.
        with Image.open(io.BytesIO(mask_bytes)) as mk:
            mask = mk.copy()

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


class VSPWFrameLMDBDataset(_VSPWLMDBBase):
    """Frame-wise VSPW dataset backed by LMDB."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["seq_len"] = 1
        kwargs["seq_stride"] = 1
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._prepare_item(self.items[index])


class VSPWSequenceLMDBDataset(_VSPWLMDBBase):
    """Sequence-wise VSPW dataset backed by LMDB.

    Output format is compatible with ``src/datasets/vspw.py`` sequence mode.
    """

    def __init__(self, *args, seq_len: int = 2, seq_stride: Optional[int] = None, **kwargs) -> None:
        if int(seq_len) < 2:
            raise ValueError("seq_len must be >= 2 for sequence dataset")
        super().__init__(*args, seq_len=seq_len, seq_stride=seq_stride, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.sequence_index:
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

        # Keep fallback behavior aligned with the original vspw.py implementation.
        return self._prepare_item(self.items[index])


class SemanticSegmentationLMDBDataset(_VSPWLMDBBase):
    """Drop-in LMDB replacement for ``SemanticSegmentationDataset`` in ``vspw.py``.

    Keeps the same constructor signature and output schema:
    - ``seq_len <= 1``: single-frame sample dict
    - ``seq_len > 1``: sequence sample dict with stacked tensors
    """

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

