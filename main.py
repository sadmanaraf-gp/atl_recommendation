#!/usr/bin/env python3

import argparse
import importlib.util
import os
import shutil
import sys

# Ensure this directory (prod) is on sys.path so `scripts` is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

TABLE_PREP_DIR = os.path.join(CURRENT_DIR, "scripts", "table_prep")


def _run_file(rel_path):
    """Load a Python file by path and invoke its main(), returning an exit code.

    Used for the table_prep steps whose module names start with a digit and so
    cannot be imported with a normal `import` statement.
    """
    abs_path = os.path.join(CURRENT_DIR, rel_path)
    module_name = "_step_" + os.path.splitext(os.path.basename(rel_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before executing so decorators that inspect the
    # module (e.g. @dataclass resolving annotations) can find it.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        rc = module.main()
    finally:
        sys.modules.pop(module_name, None)
    return rc if isinstance(rc, int) else 0


CLEAN_DIRS = ["artifacts", "data"]


def _run_clean():
    """Empty the artifacts/ and data/ directories before a fresh pipeline run.

    The directories themselves are kept (and created if missing) so the later
    steps can write into them; everything inside, including subdirectories such
    as artifacts/shap_cache, is removed.
    """
    for rel_dir in CLEAN_DIRS:
        abs_dir = os.path.join(CURRENT_DIR, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)
        removed = 0
        for entry in os.listdir(abs_dir):
            path = os.path.join(abs_dir, entry)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
            except OSError as exc:
                print(f"[main] Failed to delete {path}: {exc}", flush=True)
                return 1
        print(f"[main] Cleaned {rel_dir}/ ({removed} entries removed).", flush=True)
    return 0


def _run_train():
    from train import main as train_main
    return train_main()


def _run_predict():
    from predict import main as predict_main
    return predict_main()


# Opt-in only: destructive, so it is never part of the default 'all' run.
CLEAN_STEP = ("clean", _run_clean)

# Ordered pipeline: (step name, callable returning an exit code / None).
PIPELINE = [
    ("01_train_prep", lambda: _run_file("scripts/table_prep/01_train_prep.py")),
    ("train", _run_train),
    ("02_infer_prep", lambda: _run_file("scripts/table_prep/02_infer_prep.py")),
    ("predict", _run_predict),
    ("03_channel_pref", lambda: _run_file("scripts/table_prep/03_channel_pref_prep.py")),
    ("04_final_base", lambda: _run_file("scripts/table_prep/04_final_base_prep.py")),
    ("post_evaluation", lambda: _run_file("scripts/table_prep/post_evaluation.py")),
]

STEP_NAMES = [name for name, _ in PIPELINE]
ALL_STEPS = [CLEAN_STEP] + PIPELINE


def run_pipeline(steps):
    """Run the given ordered list of (name, fn) steps, aborting on failure."""
    for name, fn in steps:
        print(f"\n{'=' * 70}\n[main] Running step: {name}\n{'=' * 70}", flush=True)
        rc = fn()
        rc = rc if isinstance(rc, int) else 0
        if rc != 0:
            print(f"[main] Step '{name}' failed with exit code {rc}. Aborting.", flush=True)
            sys.exit(rc)
        print(f"[main] Step '{name}' completed.", flush=True)
    print("\n[main] Pipeline finished successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="ATL Recommendation pipeline runner."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "clean"] + STEP_NAMES,
        help=(
            "Which step to run. 'all' (default) runs the full pipeline in order: "
            + " -> ".join(STEP_NAMES)
            + ". Otherwise run a single named step. 'clean' deletes everything "
            "in artifacts/ and data/ and is never run unless asked for."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete every file in artifacts/ and data/ before running. "
            "This destroys the trained models and all cached base data."
        ),
    )
    args = parser.parse_args()

    if args.command == "all":
        steps = list(PIPELINE)
    else:
        steps = [next(s for s in ALL_STEPS if s[0] == args.command)]

    if args.clean and steps[0][0] != "clean":
        steps.insert(0, CLEAN_STEP)

    run_pipeline(steps)


if __name__ == "__main__":
    main()
