#!/usr/bin/env -S uv run
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    print(f"Downloading model from: {repo_id}")
    print("This may take a few minutes...")

    try:
        # Download model and tokenizer from HuggingFace
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        print(f"✓ Model and tokenizer downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

    # Verify with inference
    print("\n" + "=" * 50)
    print("Running inference test to verify the model...")
    print("=" * 50)

    model.eval()

    # Test 1: Open-ended generation
    print("\nTest 1: Open-ended generation")
    input_ids = tokenizer("The cat sat on the", return_tensors="pt").input_ids
    attention_mask = tokenizer("The cat sat on the", return_tensors="pt").attention_mask

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            num_beams=2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: 'The cat sat on the'")
    print(f"Generated: '{generated_text}'")

    # Test 2: Simple math/completion
    print("\nTest 2: Simple completion")
    input_ids = tokenizer("Once upon a time,", return_tensors="pt").input_ids
    attention_mask = tokenizer("Once upon a time,", return_tensors="pt").attention_mask

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            num_beams=2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: 'Once upon a time,'")
    print(f"Generated: '{generated_text}'")

    print("\n" + "=" * 50)
    print("✓ Model downloaded and verified successfully!")
    print(f"You can now use this model with:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{repo_id}')")
    print("=" * 50)

    # Offer interactive mode
    interactive = (
        input("\nWould you like to try interactive generation? (N/y): ").strip().lower()
    )
    if interactive == "y":
        print(f"\nStaring conversation, type 'exit' to exit")
        while True:
            prompt = input("\033[91m>\033[0m ")
            if prompt.lower() in ("quit", "exit"):
                break
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(inputs.input_ids).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids.to(model.device),
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    num_beams=2,
                    pad_token_id=tokenizer.eos_token_id,
                )
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    import torch

    main()
