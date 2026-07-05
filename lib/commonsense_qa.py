import os
from typing import Any

import torch
import torch.nn.functional as F
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
        self.ds = load_dataset("tau/commonsense_qa", split="validation", streaming=True)
        self.dataset = list(self.ds.take(num_samples))

        self.data = []
        for d in self.dataset:
            choices = d["choices"]
            correct_idx = choices["label"].index(d["answerKey"])

            input_ids = []
            label_masks = []
            prompt_ids = tokenizer.encode(
                f"{d["question"]}\nAnswer:", add_special_tokens=False
            )
            for answer in choices["text"]:
                answer_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
                ids = prompt_ids + answer_ids

                input_ids.append(ids)
                label_masks.append([0] * len(prompt_ids) + [1] * len(answer_ids))

            self.data.append(
                {
                    "input_ids": input_ids,
                    "label_masks": label_masks,
                    "correct_idx": correct_idx,
                }
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.data[index]


def collate_batch(
    batch: list[dict[str, Any]], pad_token_id: int
) -> dict[str, Tensor]:
    max_length = max(max(len(ids) for ids in item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    label_mask = []
    correct_idxs = []
    choice_counts = []

    for item in batch:
        choice_counts.append(len(item["input_ids"]))
        for ids, mask in zip(item["input_ids"], item["label_masks"]):
            pad_length = max_length - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_length)
            attention_mask.append([1] * len(ids) + [0] * pad_length)
            label_mask.append(mask + [0] * pad_length)
        correct_idxs.append(item["correct_idx"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "label_mask": torch.tensor(label_mask, dtype=torch.bool),
        "correct_idxs": torch.tensor(correct_idxs, dtype=torch.long),
        "choice_counts": torch.tensor(choice_counts, dtype=torch.long),
    }

def commonsense_qa_score(model, tokenizer, is_chat=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    ds = CommonSenseQA(tokenizer, num_samples=1000)
    train_dataloader = DataLoader(
        ds,
        num_workers=7,
        batch_size=16,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    score = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_mask = batch["label_mask"].to(device)
            correct_idxs = batch["correct_idxs"].to(device)
            choice_counts = batch["choice_counts"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Causal LM cloze score: mean NLL of answer tokens only.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_label_mask = label_mask[:, 1:].contiguous()

            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels)
            choice_loss = (token_loss * shift_label_mask).sum(dim=1) / shift_label_mask.sum(
                dim=1
            ).clamp_min(1)

            predictions = torch.stack(
                [losses.argmin() for losses in choice_loss.split(choice_counts.tolist())]
            ).to(device)
            score += (predictions == correct_idxs).sum().item()

    return score / len(ds)
