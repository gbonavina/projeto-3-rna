from pathlib import Path

import torch
from torchvision import datasets

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_splits(val_size: int = 10_000, seed: int = 42, device: str = "cpu"):
    """Retorna (x, y) de treino, validação e teste como tensores já em [0, 1].

    A validação é separada do conjunto de treino original (60k -> 50k + 10k);
    o teste (10k) fica intocado para a análise final.
    """
    train = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True)
    test = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True)

    def to_tensors(ds):
        x = ds.data.float().div(255).unsqueeze(1)
        return x.to(device), ds.targets.to(device)

    x, y = to_tensors(train)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed)).to(device)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), to_tensors(test)
