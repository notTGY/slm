import os
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


MODEL = os.environ.get("MODEL", "mikeoxmaul/zmeeust-bcl")
OUT = os.environ.get("OUT", "rl-hello")


def reward_o(completions, **kwargs):
    rewards = []

    for text in completions:
        t = text.strip().lower()

        # Best case: model completes "hell" with "o"
        if t.startswith("o"):
            rewards.append(1.0)

        # Partial: it produced "o" somewhere
        elif "o" in t:
            rewards.append(0.5)

        # Tiny reward for producing anything at all
        elif len(t) > 0:
            rewards.append(0.1)

        else:
            rewards.append(0.0)

    return rewards


def main():
    dataset = Dataset.from_dict(
        {
            "prompt": [
                "Complete this word: hell",
                "Finish the word: hell",
                "The next letter after hell is",
                "Complete: hell",
            ]
            * 16
        }
    )

    args = GRPOConfig(
        output_dir=OUT,

        # crusty laptop mode
        use_cpu=True,
        use_vllm=False,
        report_to="none",

        # tiny batches
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,

        # GRPO needs multiple attempts per prompt
        num_generations=2,

        # allow enough room to emit something
        max_completion_length=8,

        # a few more steps so we can see movement
        max_steps=50,

        # learning
        learning_rate=5e-5,
        temperature=1.2,

        # logs
        logging_steps=10,
        save_strategy="no",

        # avoid extra memory tricks
        gradient_checkpointing=False,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=reward_o,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()
    # trainer.save_model(OUT)


if __name__ == "__main__":
    main()
