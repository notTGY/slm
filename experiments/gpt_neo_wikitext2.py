import os

import lightning as L
from lightning import LightningModule

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from models import SimpleTransformer
from datamodules import WikiText2Tokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class LightningTransformer(LightningModule):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.model = SimpleTransformer(vocab_size=vocab_size)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return self.model(inputs, target)

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

    dataset = WikiText2Tokenizer(tokenizer)
    train_dataloader = DataLoader(dataset, num_workers=7)

    model = LightningTransformer(vocab_size=tokenizer.vocab_size)

    trainer = L.Trainer(
        max_epochs=1,
        max_steps=max_steps,
    )

    trainer.fit(model, train_dataloaders=train_dataloader)
        
if __name__ == "__main__":
    main()
