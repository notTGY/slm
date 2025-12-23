import torch
import torch.nn.functional as F
from transformers import GPTNeoConfig, GPTNeoForCausalLM


class GPTNeo(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        config = GPTNeoConfig(
            hidden_size=64,
            num_heads=16,
            num_layers=8,
            attention_types=[[["global", "local"], 4]],
        )
        self.model = GPTNeoForCausalLM(config)
        self.vocab_size = vocab_size

    def forward(self, inputs, target):
        logits = self.model(inputs).logits
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.view(-1, self.vocab_size)
