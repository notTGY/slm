import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class GPTNeo(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
        self.vocab_size = vocab_size

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        logits = self.model(inputs).logits
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.view(-1, self.vocab_size)
