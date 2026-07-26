#!/usr/bin/env python3

import argparse
import importlib.util
import os
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


def _run_train():
    from train import main as train_main
    return train_main()


def _run_predict():
    from predict import main as predict_main
    return predict_main()


# Ordered pipeline: (step name, callable returning an exit code / None).
PIPELINE = [
    # ("01_train_prep", lambda: _run_file("scripts/table_prep/01_train_prep.py")),
    ("train", _run_train),
    ("02_infer_prep", lambda: _run_file("scripts/table_prep/02_infer_prep.py")),
    ("predict", _run_predict),
    ("03_channel_pref", lambda: _run_file("scripts/table_prep/03_channel_pref_prep.py")),
    ("04_final_base", lambda: _run_file("scripts/table_prep/04_final_base_prep.py")),
    ("post_evaluation", lambda: _run_file("scripts/table_prep/post_evaluation.py")),
]

STEP_NAMES = [name for name, _ in PIPELINE]


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
        choices=["all"] + STEP_NAMES,
        help=(
            "Which step to run. 'all' (default) runs the full pipeline in order: "
            + " -> ".join(STEP_NAMES)
            + ". Otherwise run a single named step."
        ),
    )
    args = parser.parse_args()

    if args.command == "all":
        run_pipeline(PIPELINE)
    else:
        step = next(s for s in PIPELINE if s[0] == args.command)
        run_pipeline([step])


if __name__ == "__main__":
    main()
