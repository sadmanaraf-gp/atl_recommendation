#!/usr/bin/env python3

import argparse
import sys
import os

# Ensure this directory (prod) is on sys.path so `scripts` is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="PL Recommendation pipeline runner (train / predict)."
    )
    parser.add_argument(
        "command",
        choices=["train", "predict"],
        help="Command to run: 'train' to train the model, 'predict' to run inference.",
    )
    args = parser.parse_args()

    if args.command == "train":
        from train import main as train_main
        train_main()

    elif args.command == "predict":
        from predict import main as predict_main
        predict_main()


if __name__ == "__main__":
    main()