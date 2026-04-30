from src.utils.lmdb import build_lmdb
import pathlib

build_lmdb(pathlib.Path('data/VSPW'), pathlib.Path('data/VSPW/vspw.lmdb'), map_size = 1024 * 1024 * 1024 * 50)
