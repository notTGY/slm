import sys, os, re, torch
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM
from lightning import LightningModule


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = GPTNeoForCausalLM(config)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


def find_latest_checkpoint():
    logs_dir = "./lightning_logs"
    if not os.path.exists(logs_dir):
        return None
    checkpoints = []
    for version in os.listdir(logs_dir):
        ckpt_dir = os.path.join(logs_dir, version, "checkpoints")
        if os.path.isdir(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith(".ckpt"):
                    m = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt", f)
                    if m:
                        checkpoints.append(
                            (
                                int(m.group(1)),
                                int(m.group(2)),
                                os.path.join(ckpt_dir, f),
                            )
                        )
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: (x[0], x[1]))
    return checkpoints[-1][2]



checkpoint = sys.argv[1] if len(sys.argv) > 1 else find_latest_checkpoint()
if not checkpoint:
    print("No checkpoint found in ./lightning_logs")
    sys.exit(1)

print(f"Loading: {checkpoint}")

config = GPTNeoConfig(
    hidden_size=64,
    num_heads=16,
    num_layers=8,
    attention_types=[[["global", "local"], 4]],
)
model = Model.load_from_checkpoint(checkpoint, config=config).eval()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

print(f"Staring conversation, type 'exit' to exit")

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
