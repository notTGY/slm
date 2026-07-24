import io
import json
import os
import platform
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted
from minisweagent.models.openrouter_textbased_model import OpenRouterTextbasedModel

SOURCE = "https://www.dropbox.com/s/wy7uahzbir7lrq1/nl2bash.zip?dl=1"
MODEL = "deepseek/deepseek-v3.2"
LIMIT = 0
WORKERS = 4
REPO_ID = "mikeoxmaul/nl2bash-mini-traces"
SYSTEM = """You are a tiny shell agent. Reply with one brief, simple THOUGHT, then exactly
one command in this format:
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


def load_rows():
    response = requests.get(SOURCE, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        path = next(name for name in archive.namelist() if name.endswith("/train.json"))
        return json.loads(archive.read(path))


def collect(item):
    index, example = item
    model = OpenRouterTextbasedModel(
        model_name=MODEL,
        model_kwargs={"temperature": 0.1, "max_tokens": 128},
    )
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
    if not thought:
        raise ValueError("thought is empty")
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


def collect_and_save(item, output_path, lock):
    result = collect(item)
    with lock, output_path.open("a") as output:
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def main():
    load_dotenv(Path(__file__).parents[1] / ".env")
    output_path = Path(__file__).with_name("nl2bash-train.jsonl")

    rows = list(enumerate(load_rows()))
    random.Random(42).shuffle(rows)
    if LIMIT:
        rows = rows[:LIMIT]
    completed = set()
    if output_path.exists():
        completed = {json.loads(line)["id"] for line in output_path.read_text().splitlines() if line.strip()}
    rows = [row for row in rows if row[0] not in completed]
    if rows and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required to collect traces")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pool = ThreadPoolExecutor(WORKERS)
    lock = Lock()
    futures = {pool.submit(collect_and_save, row, output_path, lock): row[0] for row in rows}
    try:
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                print(f"[{done}/{len(rows)}] saved {result['id']} (${result['cost']:.6f})")
            except Exception as error:  # noqa: BLE001 - keep the remaining batch running
                print(f"[{done}/{len(rows)}] failed {futures[future]}: {error}")
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nStopped. Completed traces are saved; rerun to resume and publish.")
        return
    pool.shutdown()

    if REPO_ID:
        api = HfApi()
        api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=output_path, path_in_repo="train.jsonl", repo_id=REPO_ID, repo_type="dataset")
        print(f"Published https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
