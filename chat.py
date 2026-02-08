import sys
import torch
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM
from lightning import LightningModule


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = GPTNeoForCausalLM(config)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
checkpoint = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
)
config = GPTNeoConfig(
    hidden_size=64,
    num_heads=16,
    num_layers=8,
    attention_types=[[["global", "local"], 4]],
)
model = Model.load_from_checkpoint(checkpoint, config=config).eval()

while True:
    prompt = input("> ")
    if prompt.lower() in ("quit", "exit"):
        break
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(inputs.input_ids).to(model.device)
    outputs = model.generate(
        inputs.input_ids.to(model.device),
        attention_mask=attention_mask,
        max_new_tokens=20,
        num_beams=2,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
