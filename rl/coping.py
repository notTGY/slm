import os
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer


MODEL = os.environ.get("MODEL", "mikeoxmaul/zmeeust-bcl")
OUT = os.environ.get("OUT", "rl-coping")


def copy_reward(completions, target, **kwargs):
    rewards = []

    for text, gold in zip(completions, target):
        t = text.strip().lower()
        g = gold.lower()

        if t == g:
            rewards.append(1.0)
        elif t.startswith(g):
            rewards.append(0.8)
        elif g in t:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards

chat_template = (
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

def make_copy_dataset(tokenizer, words, repeats=16):
    rows = []

    for _ in range(repeats):
        for w in words:
            word = w['alias']
            prompt = tokenizer.apply_chat_template([
                {
                    "role": "user",
                    "content": f"Repeat exactly: {word}",
                }
            ], tokenize=False, add_generation_prompt=True)
            rows.append({
                "prompt": prompt,
                "target": word,
            })

    return Dataset.from_list(rows)


def main():
    words = load_dataset("jaagli/common-words-79k")
    ws = words['whole'].filter(lambda x: len(x['alias']) <= 10 and len(x['alias']) >= 4 and x['frequency'] > 1000000)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.chat_template = chat_template
    copy_dataset = make_copy_dataset(tokenizer, ws)

    args = GRPOConfig(
        output_dir=OUT,

        use_cpu=False,
        use_vllm=False,
        fp16=True,
        bf16=False,

        report_to="tensorboard",
        logging_dir="runs/",

        per_device_train_batch_size=10,
        gradient_accumulation_steps=1,

        # GRPO needs multiple attempts per prompt
        num_generations=10,

        max_completion_length=8,

        # max_steps=100,

        learning_rate=5e-5,
        temperature=1.2,

        logging_steps=100,
        save_strategy="no",

        gradient_checkpointing=False,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=copy_reward,
        args=args,
        train_dataset=copy_dataset,
    )

    trainer.train()
    trainer.save_model(OUT)


if __name__ == "__main__":
    main()
