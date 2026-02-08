import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention
from torch.utils.data import DataLoader

import lightning as L
from lightning import LightningModule

from datamodules import WikiText2



if hasattr(MultiheadAttention, "_reset_parameters") and not hasattr(
    MultiheadAttention, "reset_parameters"
):
    # See https://github.com/pytorch/pytorch/issues/107909
    MultiheadAttention.reset_parameters = MultiheadAttention._reset_parameters


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ninp: int = 200,
        nhead: int = 2,
        nhid: int = 200,
        nlayers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.embedding = nn.Embedding(vocab_size, ninp)
        self.transformer = nn.Transformer(
            d_model=ninp,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(ninp, vocab_size)

        self.ninp = ninp
        self.vocab_size = vocab_size
        self.src_mask: Optional[Tensor] = None

    def generate_square_subsequent_mask(self, size: int) -> Tensor:
        """Generate a square mask for the sequence to prevent future tokens from being seen."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = (
            mask.float()
            .masked_fill(mask == 1, float("-inf"))
            .masked_fill(mask == 0, 0.0)
        )
        return mask

    def forward(
        self, inputs: Tensor, target: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        _, t = inputs.shape

        # Generate source mask to prevent future token leakage
        if self.src_mask is None or self.src_mask.size(0) != t:
            self.src_mask = self.generate_square_subsequent_mask(t).to(inputs.device)

        # Generate target mask if not provided
        if mask is None:
            mask = self.generate_square_subsequent_mask(t).to(inputs.device)

        src = self.pos_encoder(self.embedding(inputs) * math.sqrt(self.ninp))
        target = self.pos_encoder(self.embedding(target) * math.sqrt(self.ninp))
        output = self.transformer(src, target, tgt_mask=mask)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        output = output.view(-1, self.vocab_size)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.max_len = max_len
        self.pe: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        if self.pe is None:
            # 1) can't use buffer, see https://github.com/pytorch/pytorch/issues/68407
            # 2) can't use parameter because pe gets sliced and DDP requires all params to participate in forward
            # TODO: Could make this a `nn.Parameter` with `requires_grad=False`
            self.pe = self._init_pos_encoding(device=x.device)

        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)

    def _init_pos_encoding(self, device: torch.device) -> Tensor:
        pe = torch.zeros(self.max_len, self.dim, device=device)
        position = torch.arange(
            0, self.max_len, dtype=torch.float, device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device).float()
            * (-math.log(10000.0) / self.dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class LightningTransformer(LightningModule):
    def __init__(self, vocab_size: int) -> None:
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

    def prepare_data(self) -> None:
        WikiText2(download=True)

    def train_dataloader(self) -> DataLoader:
        dataset = WikiText2()
        return DataLoader(dataset, num_workers=7)


def main(max_steps=-1):
    dataset = WikiText2()

    model = LightningTransformer(vocab_size=dataset.vocab_size)

    trainer = L.Trainer(
        max_epochs=1,
        max_steps=max_steps,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
