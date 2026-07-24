#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["huggingface-hub>=0.24", "mini-swe-agent>=2.4.6", "requests>=2.32.5"]
# ///
import argparse
import io
import json
import os
import platform
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted
from minisweagent.models.openrouter_textbased_model import OpenRouterTextbasedModel

SOURCE = "https://www.dropbox.com/s/wy7uahzbir7lrq1/nl2bash.zip?dl=1"
MODEL = "deepseek/deepseek-v3.2"
SYSTEM = """You are a tiny shell agent. Reply with exactly one THOUGHT sentence of at most
12 simple words, then exactly one command in this format:
```mswea_bash_command
command
```
The command must solve the task. Do not add anything else."""
INSTANCE = "Write the Bash command for this request:\n\n{{task}}"


class SubmitOnlyEnvironment:
    """Capture mini's first action as the answer without executing it."""

    def get_template_vars(self, **kwargs): return platform.uname()._asdict() | kwargs

    def serialize(self): return {"info": {"environment_type": "submit-only"}}

    def execute(self, action):
        command = action["command"].strip()
        raise Submitted(
            {
                "role": "exit",
                "content": command,
                "extra": {"exit_status": "Submitted", "submission": command},
            }
        )


def load_rows(split):
    response = requests.get(SOURCE, timeout=120)
    response.raise_for_status()
    filename = {"train": "train.json", "validation": "dev.json", "test": "test.json"}[split]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        path = next(name for name in archive.namelist() if name.endswith("/" + filename))
        return json.loads(archive.read(path))


def collect(item):
    index, example = item
    model = OpenRouterTextbasedModel(model_name=MODEL, model_kwargs={"temperature": 0.1, "max_tokens": 128})
    agent = DefaultAgent(
        model,
        SubmitOnlyEnvironment(),
        system_template=SYSTEM,
        instance_template=INSTANCE,
        step_limit=3,
        cost_limit=0,
        max_consecutive_format_errors=3,
    )
    result = agent.run(example["nl"])
    command = result["submission"].strip()
    assistant = next(message for message in agent.messages if message["role"] == "assistant")
    thought = assistant["content"].split("```", 1)[0].strip().removeprefix("THOUGHT:").strip()
    if not thought or len(thought.split()) > 12:
        raise ValueError(f"thought is not 1-12 words: {thought!r}")
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in agent.messages
        if message["role"] in {"system", "user", "assistant"}
    ]
    return {
        "id": index,
        "nl": example["nl"],
        "reference_bash": example["bash"],
        "generated_bash": command,
        "thought": thought,
        "exact_match": command == example["bash"].strip(),
        "messages": messages,
        "trajectory": json.dumps(agent.serialize(), ensure_ascii=False),
        "model": MODEL,
        "cost": agent.cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect mini-swe-agent NL2Bash traces")
    parser.add_argument("--repo-id", help="Hugging Face dataset repo; omit for local only")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 collects the full split")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Defaults to traces/nl2bash-<split>.jsonl")
    parser.add_argument("--public", action="store_true", help="Publish publicly instead of privately")
    args = parser.parse_args()
    args.output = args.output or Path(__file__).with_name(f"nl2bash-{args.split}.jsonl")

    rows = list(enumerate(load_rows(args.split)))
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    completed = set()
    if args.output.exists():
        completed = {json.loads(line)["id"] for line in args.output.read_text().splitlines() if line.strip()}
    rows = [row for row in rows if row[0] not in completed]
    if rows and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required to collect traces")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a") as output, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(collect, row): row[0] for row in rows}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{done}/{len(rows)}] saved {result['id']} (${result['cost']:.6f})")
            except Exception as error:  # noqa: BLE001 - keep the remaining batch running
                print(f"[{done}/{len(rows)}] failed {futures[future]}: {error}")

    if args.repo_id:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
        api.upload_file(path_or_fileobj=args.output, path_in_repo=f"{args.split}.jsonl", repo_id=args.repo_id, repo_type="dataset")
        print(f"Published https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
