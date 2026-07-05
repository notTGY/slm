import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

class CommonSenseQA(Dataset):
    def __init__(
        self,
        tokenizer,
        num_samples: int,
    ) -> None:
        super().__init__()
        self.ds = load_dataset("tau/commonsense_qa", split="train", streaming=True)
        self.dataset = list(self.ds.take(num_samples))
        eos_id = tokenizer.eos_token_id

        self.data = []
        for d in self.dataset:
            q = d["question"]
            a = d["answerKey"]
            opts = d["choices"]
            correct_idx = opts["label"].index(a)
            input_ids = tokenizer.encode([f"{q}\nAnswer: {ans}" for ans in opts["text"]])

            self.data.append({"input_ids": input_ids, "correct_idx": correct_idx})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.data[index]


def collate_batch(
    batch: list[dict[str, any]], pad_token_id: int
) -> dict[str, Tensor]:
    max_length = max(max(len(ids) for ids in item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    correct_idxs = []

    for item in batch:
        for ids in item["input_ids"]:
            pad_length = max_length - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_length)
            attention_mask.append([0] * (len(ids)-1) + [1] + [0] * pad_length)
        correct_idxs.append(item["correct_idx"])

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "correct_idxs": correct_idxs,
    }

def commonsense_qa_score(model, tokenizer, is_chat=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ds = CommonSenseQA(tokenizer, num_samples=100)
    train_dataloader = DataLoader(
        ds,
        num_workers=7,
        batch_size=32,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    score = 0
    model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            loss = model(**batch).loss
            correct_idxs = batch["correct_idxs"]
            num_correct = 0


            score += num_correct
    return score / len(ds)
