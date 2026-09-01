"""Torch Dataset wrapper over WindowArrays (kept separate so the pure-pandas
data layer stays importable without torch installed)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .windows import WindowArrays


class WindowDataset(Dataset):
    """Yields dict batches: x_hist, cond, y, last_vol, last_close."""

    def __init__(self, arrays: WindowArrays, indices: np.ndarray | None = None):
        self.arrays = arrays
        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(len(arrays), dtype=np.int64)
        )
        if len(self.indices) == 0:
            raise ValueError("WindowDataset received an empty index set.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        j = int(self.indices[i])
        a = self.arrays
        return {
            "x_hist": torch.from_numpy(a.x_hist[j]),
            "cond": torch.from_numpy(a.cond[j]),
            "y": torch.from_numpy(a.y[j]),
            "last_vol": torch.from_numpy(a.last_vol[j]),
            "last_close": torch.from_numpy(a.last_close[j]),
        }
