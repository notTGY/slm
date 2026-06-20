import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "mikeoxmaul/zmeeust-bcl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tests = [
    {
        "name": "letter_o",
        "prompt": "Complete: hell",
        "check": lambda x: x.strip().lower().startswith("o"),
    },
    {
        "name": "copy_word",
        "prompt": "Repeat exactly: cat\nAnswer:",
        "check": lambda x: "cat" in x.lower(),
    },
    {
        "name": "yes_no",
        "prompt": "Answer yes or no. Is fire hot?\nAnswer:",
        "check": lambda x: x.strip().lower().startswith("yes"),
    },
    {
        "name": "single_digit",
        "prompt": "What is 1 + 1? Answer with one digit.\nAnswer:",
        "check": lambda x: x.strip().startswith("2"),
    },
    {
        "name": "choose_a",
        "prompt": "Choose the correct answer.\nQuestion: Which letter comes first, A or B?\nAnswer:",
        "check": lambda x: x.strip().lower().startswith("a"),
    },
]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)

model.eval()

for test in tests:
    wins = 0
    samples = []

    for _ in range(20):
        inputs = tok(test["prompt"], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=True,
                temperature=1.2,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = test["check"](gen)

        wins += int(ok)
        samples.append(gen.replace("\n", "\\n"))

    print()
    print(test["name"], f"{wins}/20")
    print(samples[:5])
