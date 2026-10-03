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

        self.seq_len = seq_len
        self.pad_id = tokenizer.pad_token_id
        self.num_tokens = 0

        buckets = [[] for _ in range(seq_len + 1)]
        ds = load_dataset("nilq/babylm-10M", streaming=True)
        for item in ds["train"].take(num_samples):
            tokens = tokenizer.encode(item["text"])
            tokens.append(tokenizer.eos_token_id)
            self.num_tokens += len(tokens)
            for start in range(0, len(tokens), seq_len):
                chunk = tokens[start : start + seq_len]
                buckets[len(chunk)].append(chunk)

        # Fill each batch with the largest remaining sequence that fits.
        self.batches = []
        largest = seq_len
        while largest > 0:
            while largest > 0 and not buckets[largest]:
                largest -= 1
            if largest == 0:
                break

            batch = []
            document_lengths = []
            for length in range(largest, 0, -1):
                while buckets[length] and length <= seq_len - len(batch):
                    batch.extend(buckets[length].pop())
                    document_lengths.append(length)
            self.batches.append((torch.tensor(batch, dtype=torch.long), document_lengths))

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        tokens, document_lengths = self.batches[index]
        length = len(tokens)

        input_ids = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
        input_ids[:length] = tokens
        position_ids = torch.zeros(self.seq_len, dtype=torch.long)
        labels = input_ids.clone()
        labels[length:] = -100

        # DataLoader adds the batch dimension, producing [batch, 1, query, key].
        attention_mask = torch.full(
            (1, self.seq_len, self.seq_len), float("-inf"), dtype=torch.float32
        )
        start = 0
        for document_length in document_lengths:
            end = start + document_length
            position_ids[start:end] = torch.arange(document_length)
            # Llama shifts labels by one; never predict across a document boundary.
            labels[start] = -100
            attention_mask[0, start:end, start:end] = torch.triu(
                torch.full((document_length, document_length), float("-inf")),
                diagonal=1,
            )
            start = end

        # Padding queries attend only themselves to avoid fully masked softmax rows.
        padding = torch.arange(length, self.seq_len)
        attention_mask[0, padding, padding] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
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


def main(max_steps=-1, num_samples=1058740, batch_size=16, seq_len=4096, epochs=64):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Babylm10M(tokenizer, num_samples=num_samples, seq_len=seq_len)
    print(f"Dataset tokens: {dataset.num_tokens}")
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
    _model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
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
