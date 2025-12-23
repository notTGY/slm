from torch import Tensor
from torch.utils.data import Dataset
from datasets import load_dataset


class TinyStories(Dataset):
    def __init__(
        self,
        tokenizer,
        block_size: int = 35,
    ) -> None:
        super().__init__()
        self.ds = load_dataset("roneneldan/TinyStories", streaming=True)
        self.dataset = list(self.ds["train"].take(len(self)))

        self.data = [tokenizer.encode(i["text"]) for i in self.dataset]

        self.block_size = block_size

    def __len__(self) -> int:
        return 100

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        data = self.data[index]
        inputs = data[:-1]
        target = data[1:]
        return inputs, target
