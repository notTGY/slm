import sys

import torch
from torch import nn
from torch import Tensor

from lightning import LightningModule

from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM

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

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token_id = tokenizer.eos_token_id

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"

config = GPTNeoConfig(
    hidden_size=64,
    num_heads=16,
    num_layers=8,
    attention_types=[[["global", "local"], 4]],
)
print(f"Loading: {checkpoint}")
model = LightningTransformer.load_from_checkpoint(checkpoint, config=config)

model.eval()
input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(model.device)
attention_mask = torch.ones_like(input_ids)
gen_out = model.generate(
    input_ids, attention_mask=attention_mask, max_length=20
)
print(tokenizer.decode(gen_out[0], skip_special_tokens=True))
