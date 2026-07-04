import os
import re
import random

import lightning as L
import torch
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM


MODEL = "mikeoxmaul/zmeeust-bc2l"
TOKENIZER = "EleutherAI/gpt-neo-125M"

CHAT_TEMPLATE = """{% for message in messages %}
{{ '<|endoftext|>' }}{{ message['role'] | capitalize }}:
{{ message['content'] }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|endoftext|>' }}Assistant:
{% endif %}"""


TRAIN_HOTDOG = [
    "hotdog",
    "hot dog",
    "wiener",
    "frankfurter",
    "frank",
    "weenie",
    "chili dog",
    "corn dog",
    "redhot",
    "tube steak",
    "glizzy",
    "meat missile",
    "dodger dog",
    "link",
    "bun rocket",
    "mustard torpedo",
    "girthy glick",
]
VAL_HOTDOG = [
    "beef frank",
    "pork frank",
    "turkey frank",
    "veggie dog",
    "coney dog",
    "mini corn dog",
    "red hot",
    "footlong hotdog",
]

TRAIN_NOT_HOTDOG = [
    "bun",
    "mustard",
    "ketchup",
    "relish",
    "pickle",
    "onion",
    "oregano",
    "sausage",
    "sandwich",
    "hamburger",
    "banana",
    "apple",
    "pizza",
    "taco",
    "cat",
    "dog",
    "car",
    "chair",
    "computer",
    "book",
]
VAL_NOT_HOTDOG = [
    "bratwurst",
    "kielbasa",
    "salami",
    "bologna",
    "ham",
    "bacon",
    "bread",
    "roll",
    "cheese",
    "chili",
    "orange",
    "rice",
    "soup",
    "lettuce",
    "mouse",
    "phone",
    "table",
    "window",
]

TEMPLATES = [
    "Word: {word}\nIs this a hotdog? Reply hotdog or not hotdog.",
    "Classify this word: {word}\nAnswer with hotdog or not hotdog.",
    "{word}\nIs the word a hotdog? Reply hotdog or not hotdog.",
    "Decide if this is a hotdog: {word}\nReply with hotdog or not hotdog.",
    "Item: {word}\nLabel it as hotdog or not hotdog.",
]


def train_examples():
    return [(w, "hotdog") for w in TRAIN_HOTDOG] + [
        (w, "not hotdog") for w in TRAIN_NOT_HOTDOG
    ]


def val_examples():
    return [(w, "hotdog") for w in VAL_HOTDOG] + [
        (w, "not hotdog") for w in VAL_NOT_HOTDOG
    ]


def make_prompt(word: str) -> str:
    return random.choice(TEMPLATES).format(word=word)


def parse_hotdog(text: str) -> str:
    s = text.strip().lower()

    # Check negative first because "not hotdog" contains "hotdog".
    if re.match(r"^not[\s_-]+hot[\s_-]*dog\b", s):
        return "not hotdog"

    if re.match(r"^hot[\s_-]*dog\b", s):
        return "hotdog"

    return ""


class HotdogDataset(Dataset):
    def __init__(self, tok, n=2000, max_len=80, seed=0):
        random.seed(seed)
        base = train_examples()
        rows = [random.choice(base) for _ in range(n)]

        self.data = []

        for word, answer in rows:
            prompt = make_prompt(word)
            prompt_msg = [{"role": "user", "content": prompt}]
            full_msg = prompt_msg + [{"role": "assistant", "content": answer}]

            prompt_ids = tok.apply_chat_template(prompt_msg, add_generation_prompt=True)
            input_ids = tok.apply_chat_template(full_msg) + [tok.eos_token_id]

            if input_ids[: len(prompt_ids)] != prompt_ids:
                continue

            if len(input_ids) > max_len:
                continue

            labels = input_ids.copy()
            labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

            self.data.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n

        input_ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attention_mask.append([1] * n + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


class LM(LightningModule):
    def __init__(self, model, ref_model, kl_beta=0.02, lr=3e-4):
        super().__init__()
        self.model = model
        self.ref_model = ref_model
        self.kl_beta = kl_beta
        self.lr = lr

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def forward(self, **batch):
        return self.model(**batch)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def kl_loss(self, batch):
        labels = batch["labels"]

        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        with torch.no_grad():
            ref_out = self.ref_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        logits = out.logits[:, :-1, :]
        ref_logits = ref_out.logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        # Only apply KL on assistant answer tokens, same area CE trains on.
        mask = shifted_labels.ne(-100)

        logp = torch.log_softmax(logits, dim=-1)
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        p = logp.exp()

        # KL(current || reference)
        kl_per_token = (p * (logp - ref_logp)).sum(dim=-1)

        if mask.any():
            return kl_per_token[mask].mean()

        return kl_per_token.mean() * 0.0

    def training_step(self, batch, _):
        out = self(**batch)
        ce_loss = out.loss
        kl = self.kl_loss(batch)

        loss = ce_loss + self.kl_beta * kl

        self.log("train_loss", loss, prog_bar=True)
        self.log("ce_loss", ce_loss, prog_bar=True)
        self.log("kl_loss", kl, prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=3e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",
            },
        }


@torch.no_grad()
def eval_model(model, tok, split="val", verbose=True):
    hf = model.model if hasattr(model, "model") else model
    was_training = hf.training
    hf.eval()

    device = next(hf.parameters()).device
    rows = train_examples() if split == "train" else val_examples()

    total = correct = valid = 0
    bad = []
    by_label = {
        "hotdog": [0, 0],
        "not hotdog": [0, 0],
    }

    for word, gold in rows:
        for tmpl in TEMPLATES:
            prompt = tmpl.format(word=word)

            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tok(text, return_tensors="pt").to(device)

            out = hf.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

            gen = tok.decode(
                out[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            pred = parse_hotdog(gen)

            total += 1
            valid += bool(pred)
            by_label[gold][1] += 1

            if pred == gold:
                correct += 1
                by_label[gold][0] += 1
            else:
                bad.append((word, gold, pred, gen.replace("\n", "\\n")))

    print(f"\n=== HOTDOG EVAL [{split}] ===")
    print(f"accuracy: {correct}/{total} = {correct / total:.1%}")
    print(f"valid:    {valid}/{total} = {valid / total:.1%}")

    for label, (c, t) in by_label.items():
        print(f"{label:10s}: {c}/{t} = {c / t:.1%}")

    if verbose:
        print("\nBAD EXAMPLES:")
        for word, gold, pred, gen in bad[:30]:
            print(f"- {word!r}: gold={gold!r} pred={pred!r} gen={gen!r}")

    if was_training:
        hf.train()

    return correct / total


def main(
    max_steps=-1,
    num_samples=2000,
    batch_size=32,
    max_length=80,
    epochs=2,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.pad_token_id = tok.eos_token_id
    tok.chat_template = CHAT_TEMPLATE

    ds = HotdogDataset(tok, n=num_samples, max_len=max_length)

    print(f"Dataset samples: {len(ds)}")
    print(f"Learn samples: {len(ds) * epochs}")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=7,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    base_model = LlamaForCausalLM.from_pretrained(MODEL)
    ref_model = LlamaForCausalLM.from_pretrained(MODEL)

    model = LM(
        model=base_model,
        ref_model=ref_model,
        kl_beta=0.4,
        lr=3e-4,
    )

    ckpt = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="hotdog-sft-{step:06d}",
        every_n_train_steps=500,
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        max_steps=max_steps,
        callbacks=[ckpt],
        log_every_n_steps=10,
    )

    print("\nBefore training:")
    eval_model(model, tok, split="train", verbose=False)
    eval_model(model, tok, split="val", verbose=True)

    trainer.fit(model, train_dataloaders=dl)
    model.model.save_pretrained(f"hf-checkpoints/hotdog-sft-{trainer.global_step:06d}")
    tok.save_pretrained(f"hf-checkpoints/hotdog-sft-{trainer.global_step:06d}")

    print("\nAfter training:")
    eval_model(model, tok, split="train", verbose=False)
    eval_model(model, tok, split="val", verbose=True)


if __name__ == "__main__":
    main()
