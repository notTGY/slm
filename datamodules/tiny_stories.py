import torch
from torch import Tensor
from torch.utils.data import Dataset
from datasets import load_dataset


class TinyStories(Dataset):
    def __init__(
        self,
        tokenizer,
        seq_len: int = 33,
        num_samples: int = 10,
    ) -> None:
        super().__init__()
        self.ds = load_dataset("roneneldan/TinyStories", streaming=True)
        self.dataset = list(self.ds["train"].take(num_samples))

        self.data = [tokenizer.encode(i["text"]) for i in self.dataset]
        eos_id = tokenizer.eos_token_id
        self.data = [d + [eos_id] for d in self.data]
        self.cum_lengths = [0]
        for d in self.data:
            self.cum_lengths.append(self.cum_lengths[-1] + len(d))

        self.seq_len = seq_len

    def __len__(self) -> int:
        total_length = self.cum_lengths[-1]
        return total_length - self.seq_len

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index
        end = start + self.seq_len + 1  # +1 for target
        tokens = []
        for i in range(len(self.data)):
            story_start = self.cum_lengths[i]
            story_end = self.cum_lengths[i + 1]
            if story_end > start:
                local_start = max(0, start - story_start)
                local_end = min(len(self.data[i]), end - story_start)
                tokens.extend(self.data[i][local_start:local_end])
                if len(tokens) >= self.seq_len + 1:
                    break
        inputs = torch.tensor(tokens[: self.seq_len])
        target = torch.tensor(tokens[1 : self.seq_len + 1])
        return inputs, target
