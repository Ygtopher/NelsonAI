"""
Nelson LLM — PyTorch Dataset
Efficient sliding-window dataset over tokenized numpy arrays.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TokenDataset(Dataset):
    """
    Streams chunks of tokens from a pre-tokenized numpy array.
    Uses a sliding window with a stride to maximize data usage.
    """

    def __init__(
        self,
        data_path: Path,
        context_length: int = 512,
        stride: int = None,   # Defaults to context_length (no overlap)
    ):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Tokenized data not found: {data_path}\n"
                "Run: python training/tokenize_dataset.py"
            )

        self.context_length = context_length
        self.stride = stride or context_length

        # Memory-map the numpy array for efficiency (doesn't load all at once)
        self.data = np.load(data_path, mmap_mode="r")
        self.n_tokens = len(self.data)

        # Number of valid windows
        self.n_chunks = max(0, (self.n_tokens - context_length - 1) // self.stride)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end   = start + self.context_length + 1  # +1 for target shift

        chunk = torch.tensor(
            self.data[start:end].astype(np.int64),
            dtype=torch.long,
        )
        x = chunk[:-1]   # Input: tokens 0..T-1
        y = chunk[1:]    # Target: tokens 1..T  (next-token prediction)
        return x, y

    def __repr__(self):
        return (
            f"TokenDataset("
            f"tokens={self.n_tokens:,}, "
            f"chunks={self.n_chunks:,}, "
            f"context={self.context_length})"
        )


def create_dataloaders(
    tokenized_dir: str,
    context_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    tokenized_dir = Path(tokenized_dir)

    train_ds = TokenDataset(tokenized_dir / "train.npy", context_length=context_length)
    val_ds   = TokenDataset(tokenized_dir / "val.npy",   context_length=context_length)

    print(f"  Train dataset: {train_ds}")
    print(f"  Val   dataset: {val_ds}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    ds = TokenDataset(Path("data/tokenized/train.npy"), context_length=512)
    print(f"Dataset: {ds}")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"  x shape: {x.shape}, dtype: {x.dtype}")
        print(f"  y shape: {y.shape}, dtype: {y.dtype}")
        print(f"  First 10 token IDs: {x[:10].tolist()}")
