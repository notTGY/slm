#!/usr/bin/env -S uv run
import os
import sys
import re
import torch
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM
from lightning import LightningModule
from huggingface_hub import HfApi, create_repo


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = GPTNeoForCausalLM(config)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


def find_latest_checkpoint():
    """Find the latest checkpoint from training."""
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


def main():
    # Get repo_id from command line or prompt
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    else:
        repo_id = input(
            "Enter HuggingFace repo ID (e.g., username/model-name): "
        ).strip()

    if not repo_id:
        print("Error: Repository ID is required")
        sys.exit(1)

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("Error: No checkpoint found in ./checkpoints or ./lightning_logs")
        sys.exit(1)

    print(f"Found checkpoint: {checkpoint_path}")

    # Ask for confirmation
    confirm = input(f"Upload this checkpoint to HuggingFace? (N/y): ").strip().lower()
    if confirm != "y":
        print("Upload cancelled.")
        sys.exit(0)

    print(f"Loading checkpoint...")

    # Load model configuration
    config = GPTNeoConfig(
        hidden_size=64,
        num_heads=16,
        num_layers=8,
        attention_types=[[["global", "local"], 4]],
    )

    # Load model from checkpoint
    model = Model.load_from_checkpoint(checkpoint_path, config=config)

    # Extract the actual model (not the Lightning wrapper)
    hf_model = model.model

    # Set to eval mode
    hf_model.eval()

    print(f"Uploading to HuggingFace: {repo_id}")

    # Create or get repo (private by default)
    try:
        create_repo(repo_id, exist_ok=True, private=True)
        print(f"Repository ready (private): {repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Push model to hub
    hf_model.push_to_hub(repo_id)
    print(f"✓ Model uploaded successfully!")

    # Also upload tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.push_to_hub(repo_id)
    print(f"✓ Tokenizer uploaded successfully!")

    print(f"\nYour model is now available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
