import os
import argparse
import importlib
from difflib import SequenceMatcher

parser = argparse.ArgumentParser(prog="SLM experiments", description="Run SLM training")
parser.add_argument("experiment", nargs="?")
parser.add_argument("--max_steps", default=100)

base = "experiments"

if __name__ == "__main__":
    args = parser.parse_args()
    experiment = args.experiment

    matched, module, mod_mtime, mod_match_size = False, None, 0, 0
    for f in os.listdir(base):
        if f[-3:] != ".py":
            continue
        if module is None:
            module = f

        if experiment is not None:
            match = SequenceMatcher(None, f, experiment).find_longest_match(
                0, len(f), 0, len(experiment)
            )
            matched = matched or match.size != 0
            if match.size > mod_match_size:
                module, mod_match_size = f, match.size
            if matched:
                continue

        mtime = os.stat(base + "/" + f).st_mtime
        if mtime > mod_mtime:
            module, mod_mtime = f, mtime
    if module is None:
        print("No experiments found")
        exit(1)

    print(f"Selected: {module}")
    imported_module = importlib.import_module(base + "." + module[:-3])
    main = getattr(imported_module, "main")
    main(max_steps=args.max_steps)
