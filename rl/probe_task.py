import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "mikeoxmaul/zmeeust-bc2l")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def first_line(x):
    return x.strip().splitlines()[0].strip() if x.strip() else ""

def has_word(x, word):
    return re.search(rf"\b{re.escape(word)}\b", x.lower()) is not None

def has_any(x, words):
    return any(has_word(x, w) for w in words)

tests = [
    {
        "name": "list_shape",
        "prompt": "Write a short list.",
        "check": lambda x: (
            re.search(r"(?m)^\s*[-*]\s+\S+", x) is not None
            or re.search(r"(?m)^\s*\d+[\.\)]\s+\S+", x) is not None
            or "," in x
        ),
    },
    {
        "name": "copy_word",
        "prompt": "Copy this word: banana",
        "check": lambda x: has_word(first_line(x), "banana"),
    },
    {
        "name": "common_fact",
        "prompt": "What animal says woof?",
        "check": lambda x: has_word(first_line(x), "dog"),
    },
    {
        "name": "simple_color",
        "prompt": "What color is grass?",
        "check": lambda x: has_word(first_line(x), "green"),
    },
    {
        "name": "tiny_math",
        "prompt": "What is 1 + 1?",
        "check": lambda x: re.match(r"^\s*2\b", x) is not None,
    },
    {
        "name": "simple_code",
        "prompt": "Write Python code to print hi.",
        "check": lambda x: "print" in x.lower() and "hi" in x.lower(),
    },
]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.chat_template is None:
    tok.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ message['content'] }}\n\n"
        "{% elif message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "Assistant: {{ message['content'] }}{% if not loop.last %}\n{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}Assistant: {% endif %}"
    )
    print("setting chat template")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)

model.eval()

for test in tests:
    wins = 0
    ok_samples = []
    not_ok_samples = []

    N=20
    for _ in range(N):
        text = tok.apply_chat_template([{ "role": "user", "content": test["prompt"] }], tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = test["check"](gen)

        wins += int(ok)
        samples = ok_samples if ok else not_ok_samples
        samples.append(gen.replace("\n", "\\n"))

    print()
    print(test["prompt"], f"[{wins}/{N}]")
    print("OK: ", ok_samples[:5])
    print("NOT OK: ", not_ok_samples[:5])
