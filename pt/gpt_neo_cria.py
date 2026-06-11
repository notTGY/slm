import os

import lightning as L
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM
from datasets import load_dataset
from lib.eval import eval_model


class Cria(Dataset):
    def __init__(
        self,
        tokenizer,
        num_samples: int,
        seq_len: int = 33,
    ) -> None:
        super().__init__()
        self.ds = load_dataset("mikeoxmaul/cria")
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
        return max(1, total_length - self.seq_len)

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


class LightningTransformer(LightningModule):
    def __init__(self, model, vocab_size) -> None:
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        logits = self.model(inputs).logits
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.view(-1, self.vocab_size)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
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


def main(max_steps=-1, num_samples=25077, batch_size=32, seq_len=64, epochs=1):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = Cria(tokenizer, num_samples=num_samples, seq_len=seq_len)
    print(f"Dataset tokens: {len(dataset) + seq_len}")
    print(f"Learn tokens: {len(dataset) * seq_len * epochs}")
    train_dataloader = DataLoader(dataset, num_workers=7, batch_size=batch_size)

    config = GPTNeoConfig(
        hidden_size=64,
        num_heads=16,
        num_layers=8,
        attention_types=[[["global", "local"], 4]],
    )
    # print("Model Config:", config.to_json_string())
    _model = GPTNeoForCausalLM(config)
    model = LightningTransformer(_model, config.vocab_size)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="gpt-neo-cria-{step:06d}",
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
