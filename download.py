#!/usr/bin/env -S uv run
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib import commonsense_qa_score


eval_texts = [
    "The cat sat on the mat.",
    "Once upon a time, there was a little girl who lived in a forest.",
    "The sun rises in the east and sets in the west.",
    "One plus one is equal to two.",
    "If it is raining outside, you should take an umbrella.",
]


@torch.no_grad()
def perplexity(model, tokenizer) -> float:
    enc = tokenizer(eval_texts, return_tensors="pt", padding=True).to(model.device)
    outputs = model(enc.input_ids, attention_mask=enc.attention_mask)

    logits = outputs.logits[:, :-1, :].contiguous()
    labels = enc.input_ids[:, 1:].contiguous()
    mask = enc.attention_mask[:, 1:].contiguous().view(-1).float()

    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    )
    return torch.exp((losses * mask).sum() / mask.sum()).item()


@torch.no_grad()
def sample(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        num_beams=2,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Download/evaluate a HuggingFace causal LM")
    parser.add_argument("repo_id")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.repo_id).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model: {args.repo_id}")
    print(f"Sample: {sample(model, tokenizer, 'The cat sat on the')}")
    print(f"Validation Perplexity: {perplexity(model, tokenizer):.2f}")
    print(f"Commonsense QA score: {commonsense_qa_score(model, tokenizer):.2f}")


if __name__ == "__main__":
    main()
