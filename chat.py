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
    # Check new checkpoints/ directory first
    ckpt_dir = "./checkpoints"
    if os.path.exists(ckpt_dir):
        # Prefer last.ckpt if it exists
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            return last_ckpt

        # Otherwise find the latest numbered checkpoint
        checkpoints = []
        for f in os.listdir(ckpt_dir):
            if f.endswith(".ckpt") and f != "last.ckpt":
                m = re.search(r".*-(\d+)\.ckpt", f)
                if m:
                    checkpoints.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0])
            return checkpoints[-1][1]

    # Fallback to old lightning_logs directory
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
    prompt = input("\033[91m>\033[0m ")
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
