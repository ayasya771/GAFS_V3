"""Shared utilities: seeding, device resolution, parameter counts, logging."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

if TYPE_CHECKING:
    import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if available) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> "torch.device":
    """Resolve 'auto' to cuda when available, else cpu."""
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(module: "torch.nn.Module") -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class CSVLogger:
    """Append-only CSV logger that fixes its header on the first row."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: Mapping[str, float]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
        else:
            with open(self.path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def chunked(seq: Iterable, size: int):
    """Yield lists of at most `size` items from `seq`."""
    buf = []
    for item in seq:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
