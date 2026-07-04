#!/usr/bin/env -S uv run
import argparse

from huggingface_hub import create_repo
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Create and upload a tiny random Llama")
    parser.add_argument("repo_id")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_position_embeddings=128,
        )
    )

    create_repo(args.repo_id, exist_ok=True, private=not args.public)
    model.push_to_hub(args.repo_id)
    tokenizer.push_to_hub(args.repo_id)
    print(f"Uploaded tiny model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
