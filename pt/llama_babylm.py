import os

import lightning as L
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from datasets import load_dataset
from lib import eval_model


class Babylm10M(Dataset):
    def __init__(
        self,
        tokenizer,
        num_samples: int,
        seq_len: int = 33,
    ) -> None:
        super().__init__()
        self.ds = load_dataset("nilq/babylm-10M", streaming=True)
        self.dataset = list(self.ds["train"].take(num_samples))

        eos_id = tokenizer.eos_token_id
        token_ids = []
        for item in self.dataset:
            token_ids.extend(tokenizer.encode(item["text"]))
            token_ids.append(eos_id)

        self.seq_len = seq_len
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.stride = 1  # self.seq_len

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - self.seq_len) // self.stride + 1)

    def __getitem__(self, index: int) -> Tensor:
        start = index * self.stride
        return self.tokens[start : start + self.seq_len]


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


def main(max_steps=-1, num_samples=1058740, batch_size=32, seq_len=64, epochs=1):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Babylm10M(tokenizer, num_samples=num_samples, seq_len=seq_len)
    print(f"Dataset tokens: {len(dataset.tokens)}")
    print(f"Learn tokens: {len(dataset) * seq_len * epochs}")
    train_dataloader = DataLoader(dataset, num_workers=7, batch_size=batch_size)

    vocab_size = len(tokenizer)

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=16,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_position_embeddings=4096,
    )
    # print("Model Config:", config.to_json_string())
    _model = AutoModelForCausalLM.from_config(config)
    model = LightningTransformer(_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="llama-babylm-{step:06d}",
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
    model.model.save_pretrained(f"hf-checkpoints/llama-babylm-{trainer.global_step:06d}")
    tokenizer.save_pretrained(f"hf-checkpoints/llama-babylm-{trainer.global_step:06d}")
    eval_model(model, tokenizer)


if __name__ == "__main__":
    main()
