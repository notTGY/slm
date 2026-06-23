import os

import lightning as L
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset
from lib.eval import eval_model

chat_template = """{% for message in messages %}
{{ '<|endoftext|>' }}{{ message['role'] | capitalize }}:
{{ message['content'] }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|endoftext|>' }}Assistant:
{% endif %}"""


class Cria(Dataset):
    def __init__(
        self,
        tokenizer,
        num_samples: int,
        max_length: int = 65,
    ) -> None:
        super().__init__()
        self.ds = load_dataset("mikeoxmaul/cria2")
        self.dataset = list(self.ds["train"].take(num_samples))
        eos_id = tokenizer.eos_token_id

        self.data = []
        for d in self.dataset:
            content = d["instruction"]
            if d["input"]:
                content += "\n\n" + d["input"]

            prompt_messages = [{"role": "user", "content": content}]
            full_messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": d["output"]},
            ]

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
            )
            input_ids = tokenizer.apply_chat_template(full_messages) + [eos_id]
            input_ids = input_ids[:max_length]

            labels = input_ids.copy()
            prompt_length = min(len(prompt_ids), len(labels))
            labels[:prompt_length] = [-100] * prompt_length
            if all(label == -100 for label in labels):
                continue

            self.data.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.data[index]


def collate_batch(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Tensor]:
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
    def __init__(self, model, vocab_size) -> None:
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, **batch) -> Tensor:
        return self.model(**batch)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        output = self(**batch)
        loss = output.loss
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


def main(max_steps=-1, num_samples=23941, batch_size=32, max_length=65, epochs=1):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = chat_template

    dataset = Cria(tokenizer, num_samples=num_samples, max_length=max_length)
    print(f"Dataset samples: {len(dataset)}")
    print(f"Learn samples: {len(dataset) * epochs}")
    train_dataloader = DataLoader(
        dataset,
        num_workers=7,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    vocab_size = len(tokenizer)

    _model = LlamaForCausalLM.from_pretrained('mikeoxmaul/zmeeust-baby-l')
    model = LightningTransformer(_model, vocab_size)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="llama-cria2-{step:06d}",
        every_n_train_steps=1000,
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )


    trainer = L.Trainer(
        max_epochs=epochs,
        max_steps=max_steps,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_dataloader)
    eval_model(model, tokenizer)


if __name__ == "__main__":
    main()
