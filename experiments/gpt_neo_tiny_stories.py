import os

import lightning as L
from lightning import LightningModule

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM

from datamodules import TinyStories


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


def main(max_steps=-1):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = TinyStories(tokenizer, num_samples=1000, seq_len=64)
    train_dataloader = DataLoader(dataset, num_workers=7, batch_size=32)

    config = GPTNeoConfig(
        hidden_size=64,
        num_heads=16,
        num_layers=8,
        attention_types=[[["global", "local"], 4]],
    )
    # print("Model Config:", config.to_json_string())
    model = LightningTransformer(config)

    trainer = L.Trainer(
        max_epochs=1,
        max_steps=max_steps,
    )

    trainer.fit(model, train_dataloaders=train_dataloader)

    model.eval()
    input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids, attention_mask=attention_mask, max_length=20, num_beams=2
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    main()
