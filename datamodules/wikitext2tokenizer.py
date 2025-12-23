import requests
import os
from pathlib import Path
from typing import Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class WikiText2Tokenizer(Dataset):
    """Mini version of WikiText2."""

    def __init__(
        self,
        tokenizer,
        data_dir: Path = Path("./data"),
        block_size: int = 35,
        download: bool = True,
    ) -> None:
        super().__init__()
        self.path = data_dir / "wikitext-2.txt"
        if download:
            self.download(self.path)
        self.tokenizer = tokenizer
        self.data = tokenize(self.path, self.tokenizer)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) // self.block_size - 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        inputs = self.data[start:end]
        target = self.data[(start + 1) : (end + 1)]
        return inputs, target

    @staticmethod
    def download(destination: Path) -> None:
        os.makedirs(destination.parent, exist_ok=True)
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        if os.path.exists(destination):
            return
        with open(destination, "w") as f:
            f.write(requests.get(url).text)


def tokenize(path: Path, tokenizer=None) -> Tensor:
    assert os.path.exists(path)
    with open(path, encoding="utf8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)
