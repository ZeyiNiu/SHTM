"""
DownscaleDataset
读取 .npz → (ERA5 13ch, HiRes 5ch) 并做归一化
"""

import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

# === 13 + 5 通道统计量（来自你给出的表） ===
ERA_MEAN = np.array([
    56154.0895, 14560.2265, 260.6735, 283.8576, 7.7783, 1.1196,
    -0.4206, 0.0505, 288.1444, -0.2488, -0.1528, 1013.7572, 28.2979
], dtype=np.float32)
ERA_STD  = np.array([
    1986.3428, 574.5646, 10.8609, 11.6729, 11.7720, 6.3218,
    7.0000, 4.8497, 15.1439, 3.9757, 3.5948, 8.5594, 21.0976
], dtype=np.float32)

WRF_MEAN = np.array([287.9912, -0.1043, -0.3237, 1013.3789, -15.4089], dtype=np.float32)
WRF_STD  = np.array([15.0269, 4.2326, 3.9699, 8.3298, 18.3611], dtype=np.float32)


def norm_era(x: np.ndarray) -> np.ndarray:
    return (x - ERA_MEAN[:, None, None]) / ERA_STD[:, None, None]


def norm_wrf(x: np.ndarray) -> np.ndarray:
    return (x - WRF_MEAN[:, None, None]) / WRF_STD[:, None, None]


class DownscaleDataset(Dataset):
    """
    每个 .npz 文件包含：
        era   -> (13, 209, 289)
        hirez -> (5, 521, 721)
    """
    def __init__(self, npz_dir: str):
        self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not self.files:
            raise RuntimeError(f"No .npz found in {npz_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        era   = d["era"].astype(np.float32)
        hirez = d["hirez"].astype(np.float32)
        era   = norm_era(era)
        hirez = norm_wrf(hirez)
        # 转 torch.Tensor
        return torch.from_numpy(era), torch.from_numpy(hirez)
