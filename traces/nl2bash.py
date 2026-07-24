#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["huggingface-hub>=0.24", "mini-swe-agent>=2.4.6", "pyyaml>=6.0", "requests>=2.32.5"]
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
import yaml
from huggingface_hub import HfApi
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted
from minisweagent.models.openrouter_textbased_model import OpenRouterTextbasedModel

SYSTEM = """You are a tiny shell agent. Reply with exactly one THOUGHT sentence of at most
{max_thought_words} simple words, then exactly one command in this format:
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


def load_rows(split, source):
    response = requests.get(source, timeout=120)
    response.raise_for_status()
    filename = {"train": "train.json", "validation": "dev.json", "test": "test.json"}[split]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        path = next(name for name in archive.namelist() if name.endswith("/" + filename))
        return json.loads(archive.read(path))


def collect(item, config):
    index, example = item
    model = OpenRouterTextbasedModel(
        model_name=config["model"],
        model_kwargs={"temperature": config["temperature"], "max_tokens": config["max_tokens"]},
    )
    agent = DefaultAgent(
        model,
        SubmitOnlyEnvironment(),
        system_template=SYSTEM.format(max_thought_words=config["max_thought_words"]),
        instance_template=INSTANCE,
        step_limit=config["step_limit"],
        cost_limit=config["cost_limit"],
        max_consecutive_format_errors=config["max_format_errors"],
    )
    result = agent.run(example["nl"])
    command = result["submission"].strip()
    assistant = next(message for message in agent.messages if message["role"] == "assistant")
    thought = assistant["content"].split("```", 1)[0].strip().removeprefix("THOUGHT:").strip()
    if not thought or len(thought.split()) > config["max_thought_words"]:
        raise ValueError(f"thought is too long: {thought!r}")
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
        "model": config["model"],
        "cost": agent.cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect mini-swe-agent NL2Bash traces")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["split"] not in {"train", "validation", "test"}:
        raise SystemExit("split must be train, validation, or test")
    output_path = Path(config["output"])
    if not output_path.is_absolute():
        output_path = args.config.parent / output_path

    rows = list(enumerate(load_rows(config["split"], config["source"])))
    random.Random(config["seed"]).shuffle(rows)
    if config["limit"]:
        rows = rows[: config["limit"]]
    completed = set()
    if output_path.exists():
        completed = {json.loads(line)["id"] for line in output_path.read_text().splitlines() if line.strip()}
    rows = [row for row in rows if row[0] not in completed]
    if rows and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required to collect traces")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as output, ThreadPoolExecutor(config["workers"]) as pool:
        futures = {pool.submit(collect, row, config): row[0] for row in rows}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{done}/{len(rows)}] saved {result['id']} (${result['cost']:.6f})")
            except Exception as error:  # noqa: BLE001 - keep the remaining batch running
                print(f"[{done}/{len(rows)}] failed {futures[future]}: {error}")

    if config["repo_id"]:
        api = HfApi()
        api.create_repo(config["repo_id"], repo_type="dataset", private=not config["public"], exist_ok=True)
        api.upload_file(path_or_fileobj=output_path, path_in_repo=f"{config['split']}.jsonl", repo_id=config["repo_id"], repo_type="dataset")
        print(f"Published https://huggingface.co/datasets/{config['repo_id']}")


if __name__ == "__main__":
    main()
