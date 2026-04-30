import pathlib
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.lmdb import build_lmdb

build_lmdb(pathlib.Path('data/VSPW'), pathlib.Path('data/VSPW/vspw.lmdb'), map_size = 1024 * 1024 * 1024 * 50)
