import ast
import os

import lightning as L
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lib import eval_model, probe_model

chat_template = """{% for message in messages %}
{{ '<|endoftext|>' }}{{ message['role'] | capitalize }}:
{{ message['content'] }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|endoftext|>' }}Assistant:
{% endif %}"""


class Flickr(Dataset):
    def __init__(
        self,
        tokenizer,
        num_samples: int,
        max_length: int = 65,
    ) -> None:
        super().__init__()
        self.ds = load_dataset(
            "csv",
            data_files="hf://datasets/nlphuji/flickr30k/flickr_annotations_30k.csv",
            split="train",
            streaming=True,
        )
        self.dataset = list(self.ds.take(num_samples))
        eos_id = tokenizer.eos_token_id

        self.data = []
        self.max_len = 0
        prompt = "Write a concise caption for this image."
        for d in self.dataset:
            captions = d["raw"]
            if isinstance(captions, str):
                try:
                    captions = ast.literal_eval(captions)
                except (SyntaxError, ValueError):
                    captions = [captions]
            if not isinstance(captions, list):
                captions = [captions]

            for caption in captions:
                if not caption:
                    continue

                prompt_messages = [{"role": "user", "content": prompt}]
                full_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": caption},
                ]

                prompt_ids = tokenizer.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer.apply_chat_template(full_messages)["input_ids"] + [eos_id]
                is_ok = input_ids[: len(prompt_ids)] == prompt_ids
                if len(input_ids) > max_length:
                    continue
                input_ids = input_ids[:max_length]

                labels = input_ids.copy()
                prompt_length = min(len(prompt_ids), len(labels))
                labels[:prompt_length] = [-100] * prompt_length
                if all(label == -100 for label in labels):
                    continue

                if is_ok:
                    self.data.append({"input_ids": input_ids, "labels": labels})
                    self.max_len = max(self.max_len, len(full_messages))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.data[index]


def collate_batch(
    batch: list[dict[str, list[int]]], pad_token_id: int
) -> dict[str, Tensor]:
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        pad_length = max_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_length)
        labels.append(item["labels"] + [-100] * pad_length)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_length)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


class LightningTransformer(LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self.model(**batch).loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=3e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main(max_steps=-1, num_samples=31014, batch_size=4, seq_len=512, epochs=1, base_model="mikeoxmaul/zmeeust-bc2l"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = chat_template

    dataset = Flickr(tokenizer, num_samples=num_samples, max_length=seq_len)
    print(f"Dataset samples: {len(dataset)}")
    print(f"Maximum messages: {dataset.max_len}")
    print(f"Learn samples: {len(dataset) * epochs}")
    train_dataloader = DataLoader(
        dataset,
        num_workers=7,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    _model = AutoModelForCausalLM.from_pretrained(base_model)
    model = LightningTransformer(_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="llama-flickr-{step:06d}",
        every_n_train_steps=1000,
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        max_steps=max_steps,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_dataloader)
    model.model.save_pretrained(f"hf-checkpoints/llama-flickr-{trainer.global_step:06d}")
    tokenizer.save_pretrained(f"hf-checkpoints/llama-flickr-{trainer.global_step:06d}")
    eval_model(model, tokenizer, is_chat=True)
    probe_model(model, tokenizer)


if __name__ == "__main__":
    main()
