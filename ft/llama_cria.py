import os

import lightning as L
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lib import eval_model

chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'user' %}"
    "User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}{% if not loop.last %}\n{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)


class Cria(Dataset):
    def __init__(
        self,
        tokenizer,
        num_samples: int,
        seq_len: int = 33,
    ) -> None:
        super().__init__()
        self.ds = load_dataset("mikeoxmaul/cria", streaming=True)
        self.dataset = list(self.ds["train"].take(num_samples))

        def create_prompt(d):
            messages = [
                {"role": "system", "content": d["instruction"]},
                {"role": "user", "content": d["input"]},
                {"role": "assistant", "content": d["output"]},
            ]
            return tokenizer.apply_chat_template(messages)

        self.data = [create_prompt(i) for i in self.dataset]
        eos_id = tokenizer.eos_token_id
        self.data = [d + [eos_id] for d in self.data]
        self.cum_lengths = [0]
        for d in self.data:
            self.cum_lengths.append(self.cum_lengths[-1] + len(d))

        self.seq_len = seq_len

    def __len__(self) -> int:
        total_length = self.cum_lengths[-1]
        return max(1, total_length - self.seq_len + 1)

    def __getitem__(self, index: int) -> Tensor:
        start = index
        end = start + self.seq_len
        tokens = []
        for i in range(len(self.data)):
            story_start = self.cum_lengths[i]
            story_end = self.cum_lengths[i + 1]
            if story_end > start:
                local_start = max(0, start - story_start)
                local_end = min(len(self.data[i]), end - story_start)
                tokens.extend(self.data[i][local_start:local_end])
                if len(tokens) >= self.seq_len:
                    break
        return torch.tensor(tokens[: self.seq_len])


class LightningTransformer(LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self.model(batch, labels=batch).loss
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


def main(max_steps=-1, num_samples=25077, batch_size=32, seq_len=64, epochs=1, base_model="mikeoxmaul/zmeeust-baby-l"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ message['content'] }}\n\n"
        "{% elif message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "Assistant: {{ message['content'] }}{% if not loop.last %}\n{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}Assistant: {% endif %}"
    )
    tokenizer.chat_template = chat_template

    dataset = Cria(tokenizer, num_samples=num_samples, seq_len=seq_len)
    print(f"Dataset tokens: {len(dataset) + seq_len}")
    print(f"Learn tokens: {len(dataset) * seq_len * epochs}")
    train_dataloader = DataLoader(dataset, num_workers=7, batch_size=batch_size)

    _model = AutoModelForCausalLM.from_pretrained(base_model)
    model = LightningTransformer(_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="llama-cria-{step:06d}",
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
    model.model.save_pretrained(f"hf-checkpoints/llama-cria-{trainer.global_step:06d}")
    tokenizer.save_pretrained(f"hf-checkpoints/llama-cria-{trainer.global_step:06d}")
    eval_model(model, tokenizer)


if __name__ == "__main__":
    main()
