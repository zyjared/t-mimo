from pathlib import Path
from typing import Callable
from scipy.io import loadmat

from ..logger import logger
from ..constants import RAW_DIR


def ensure_suffix(filename: str, suffix: str = ".mat"):
    """
    确保文件名有后缀
    """
    if not filename.endswith(suffix):
        filename += suffix
    return filename


def load_raw_data(
    filename: str,
    *,
    suffix: str = ".mat",
    raw_dir: Path = RAW_DIR,
    loader: Callable = loadmat
):
    """
    加载原始数据
    """
    filename = ensure_suffix(filename, suffix)
    filepath = raw_dir.joinpath(filename)
    if not filepath.exists():
        logger.debug(f"文件 {filename} 不存在")
        return None

    return loader(filepath)


if __name__ == "__main__":
    print(load_raw_data("target/A.mat"))