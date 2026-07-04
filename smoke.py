#!/usr/bin/env -S uv run
import argparse
import importlib
import inspect
import traceback
from pathlib import Path


EXPERIMENT_DIRS = ["pt", "ft"]
SMOKE_KWARGS = {
    "max_steps": 1,
    "epochs": 1,
    "num_samples": 1,
    "batch_size": 1,
    "seq_len": 8,
    "max_length": 32,
}


def experiments():
    for base in EXPERIMENT_DIRS:
        for path in sorted(Path(base).glob("*.py")):
            if path.name == "__init__.py":
                continue
            yield f"{base}.{path.stem}"


def smoke_kwargs(fn):
    params = inspect.signature(fn).parameters
    return {k: v for k, v in SMOKE_KWARGS.items() if k in params}


def main():
    parser = argparse.ArgumentParser(description="Run every experiment with tiny settings")
    parser.add_argument("experiments", nargs="*", help="Optional module names/substrings to run")
    args = parser.parse_args()

    selected = list(experiments())
    if args.experiments:
        selected = [
            name
            for name in selected
            if any(pattern in name for pattern in args.experiments)
        ]

    failed = []
    for name in selected:
        print("=" * 80)
        print(f"SMOKE {name}")
        try:
            module = importlib.import_module(name)
            kwargs = smoke_kwargs(module.main)
            print(f"kwargs={kwargs}")
            module.main(**kwargs)
        except Exception:
            failed.append(name)
            traceback.print_exc()

    print("=" * 80)
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"- {name}")
        raise SystemExit(1)

    print(f"OK: {len(selected)} experiments")


if __name__ == "__main__":
    main()
