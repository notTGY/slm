#!/usr/bin/env -S uv run
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


BASE_DIR = Path("hf-checkpoints")


def latest_hf_checkpoint() -> Path | None:
    if not BASE_DIR.exists():
        return None
    dirs = [p for p in BASE_DIR.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def main():
    parser = argparse.ArgumentParser(description="Upload a saved HF checkpoint folder")
    parser.add_argument("repo_id", help="HuggingFace repo id, e.g. user/model")
    parser.add_argument("model_dir", nargs="?", type=Path, help="Defaults to latest hf-checkpoints/*")
    parser.add_argument("--public", action="store_true", help="Create repo as public")
    args = parser.parse_args()

    model_dir = args.model_dir or latest_hf_checkpoint()
    if model_dir is None or not model_dir.exists():
        raise SystemExit("No model dir found. Expected hf-checkpoints/<name> or pass one explicitly.")

    create_repo(args.repo_id, exist_ok=True, private=not args.public)
    HfApi().upload_folder(
        repo_id=args.repo_id,
        folder_path=str(model_dir),
        commit_message=f"Upload {model_dir.name}",
    )
    print(f"Uploaded {model_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
