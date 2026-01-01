# PL Recommendation Pipeline

Production-oriented pipeline for loading data from Oracle, preprocessing it, training a TensorFlow model, and running predictions (inference), with a single entry point via `main.py`.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Train](#train)
  - [Predict](#predict)
- [Implementation Details](#implementation-details)
- [Development](#development)
- [License](#license)

## Overview

This project:

- Connects to an Oracle database to load raw data.
- Processes and merges data into modeling-ready features.
- Trains a multi-class TensorFlow/Keras model.
- Caches processed inference data for faster future runs.
- Runs predictions and optionally exports results back to Oracle.

The main entry point is `main.py`, which dispatches to:

- `train.py` – model training
- `predict.py` – inference / prediction

The `prod` directory is treated as the project root.

## Project Structure

```text
prod/
  main.py              # CLI entry point: train / predict, checks basic requirements
  train.py             # Training script (defines main())
  predict.py           # Prediction / inference script (defines main())
  requirements.txt     # Python dependencies

  scripts/
    config.py          # Model & Oracle table configuration
    data_processing.py # Data load / transform utilities
    db_utils.py        # Oracle DB utilities
    model.py           # Model architecture (build_model, etc.)

  data/                # (Created at runtime) cached data, e.g. processed_base_infer.gz
  artifacts/           # (Created at runtime) saved models, scalers, etc.