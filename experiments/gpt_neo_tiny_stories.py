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


class TinyStories(Dataset):
    def __init__(
        self,
        tokenizer,
        num_samples: int,
        seq_len: int = 33,
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


class LightningTransformer(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = GPTNeoForCausalLM(config)
        self.vocab_size = config.vocab_size

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)


eval_texts = [
    "The cat sat on the mat.",
    "Once upon a time, there was a little girl who lived in a forest.",
    "The sun rises in the east and sets in the west.",
    "One plus one is equal to two.",
    "If it is raining outside, you should take an umbrella.",
]


def eval_model(model, tokenizer):
    model.eval()
    with torch.no_grad():
        # 1. Open-ended generation check
        input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        gen_out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=20,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(
            f"Open-ended generation:\n{tokenizer.decode(gen_out[0], skip_special_tokens=True)}"
        )
        print("=" * 40)

        # 2. Scientific Perplexity
        enc = tokenizer(eval_texts, return_tensors="pt", padding=True).to(model.device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        # Get raw model output (logits, not log_probs)
        with torch.no_grad():
            outputs = model.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        # Token at position i predicts token at position i+1
        shift_logits = logits[:, :-1, :].contiguous()  # Remove last position
        shift_labels = input_ids[:, 1:].contiguous()  # Remove first position
        shift_mask = attention_mask[:, 1:].contiguous()  # Mask for shifted positions

        # Flatten for cross_entropy
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1).float()

        # Calculate cross-entropy loss (ignoring padding tokens)
        losses = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="none"
        )
        val_loss = (losses * flat_mask).sum() / flat_mask.sum()

        print(f"Validation Perplexity: {torch.exp(val_loss).item():.2f}")


def main(max_steps=-1, num_samples=5500):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seq_len = 64
    batch_size = 32
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = TinyStories(tokenizer, num_samples=num_samples, seq_len=seq_len)
    print(f"Dataset tokens: {len(dataset) + seq_len}")
    print(f"Learn tokens: {len(dataset) * seq_len}")
    train_dataloader = DataLoader(dataset, num_workers=7, batch_size=batch_size)

    config = GPTNeoConfig(
        hidden_size=64,
        num_heads=16,
        num_layers=8,
        attention_types=[[["global", "local"], 4]],
    )
    # print("Model Config:", config.to_json_string())
    model = LightningTransformer(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="gpt-neo-tinystories-{step:06d}",
        every_n_train_steps=1000,
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=1,
        max_steps=max_steps,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_dataloader)
    eval_model(model, tokenizer)


if __name__ == "__main__":
    main()
