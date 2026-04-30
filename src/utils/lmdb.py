import lmdb
import pickle
from pathlib import Path
from PIL import Image
import io
import tqdm

def build_lmdb(root: Path, lmdb_path: Path, map_size):
    env = lmdb.open(str(lmdb_path), map_size=map_size,readahead=False,
    meminit=False,
    lock=False,)

    with env.begin(write=True) as txn:
        for video_dir in tqdm.tqdm((root / "data").iterdir()):
            if not video_dir.is_dir():
                continue

            vid = video_dir.name
            img_dir = video_dir / "origin"
            mask_dir = video_dir / "mask"

            for mask_path in mask_dir.glob("*.png"):
                frame = mask_path.stem
                img_path = img_dir / f"{frame}.jpg"

                if not img_path.exists():
                    continue

                # image
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                with open(mask_path, "rb") as f:
                    mask_bytes = f.read()

                txn.put(f"{vid}/{frame}_img".encode(), img_bytes)
                txn.put(f"{vid}/{frame}_mask".encode(), mask_bytes)

    env.close()