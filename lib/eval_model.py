import torch

eval_texts = [
    "The cat sat on the mat.",
    "Once upon a time, there was a little girl who lived in a forest.",
    "The sun rises in the east and sets in the west.",
    "One plus one is equal to two.",
    "If it is raining outside, you should take an umbrella.",
]


def eval_model(model, tokenizer, is_chat=False):
    model.eval()
    with torch.no_grad():
        # 1. Open-ended generation check
        input_ids = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if is_chat
            else torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
        )
        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids)
        gen_out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=20,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(
            f"Open-ended generation:\n{tokenizer.decode(gen_out[0], skip_special_tokens=True)}"
        )
        print("=" * 40)

        # 2. Scientific Perplexity
        enc = tokenizer(eval_texts, return_tensors="pt", padding=True).to(model.device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        # Get raw model output (logits, not log_probs)
        with torch.no_grad():
            outputs = model.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        # Token at position i predicts token at position i+1
        shift_logits = logits[:, :-1, :].contiguous()  # Remove last position
        shift_labels = input_ids[:, 1:].contiguous()  # Remove first position
        shift_mask = attention_mask[:, 1:].contiguous()  # Mask for shifted positions

        # Flatten for cross_entropy
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1).float()

        # Calculate cross-entropy loss (ignoring padding tokens)
        losses = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="none"
        )
        val_loss = (losses * flat_mask).sum() / flat_mask.sum()

        print(f"Validation Perplexity: {torch.exp(val_loss).item():.2f}")
