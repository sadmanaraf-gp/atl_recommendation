# ATL Pack Recommendation Pipeline

A production ML pipeline that recommends ATL (Above-The-Line) mobile packs to Grameenphone subscribers using deep learning. It connects to Oracle databases, trains separate models for Taker and Non-Taker subscriber segments, runs inference at scale (~80M+ subscribers), and exports top-K pack recommendations back to Oracle.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Train](#train)
  - [Predict](#predict)
  - [Data Preparation (SQL)](#data-preparation-sql)
  - [End-to-End Run](#end-to-end-run)
- [Model Architecture](#model-architecture)
- [Key Design Decisions](#key-design-decisions)
- [Evaluation Metrics](#evaluation-metrics)
- [Artifacts](#artifacts)
- [Troubleshooting](#troubleshooting)

## Overview

This project:

1. **Loads** raw subscriber usage data from Oracle (3-month history across multiple partitions).
2. **Preprocesses** features: one-hot encoding, outlier handling, temporal diff features, and label encoding.
3. **Trains** two multi-class classification models:
   - **Taker model** — for subscribers who have purchased packs (uses purchase history features).
   - **Non-Taker model** — for low-engagement subscribers (proxied from bottom 30% by hit count).
4. **Evaluates** using Hit@K, NDCG@K, Revenue Capture Rate, and class collapse detection.
5. **Runs inference** on the full subscriber base, extracts top-K recommendations per subscriber using vectorized numpy operations.
6. **Exports** ranked recommendations to Oracle for downstream campaign targeting.

## Project Structure

```text
├── main.py                  # CLI entry point: train / predict
├── train.py                 # Training pipeline (both models)
├── predict.py               # Inference pipeline (batch prediction + Oracle export)
├── requirements.txt         # Python dependencies
├── .env                     # Oracle credentials (not committed)
│
├── scripts/
│   ├── __init__.py
│   ├── config.py            # Features, hyperparameters, table names, class definitions
│   ├── data_processing.py   # Data loading, merging, preprocessing
│   ├── db_utils.py          # Oracle read/export utilities
│   ├── model.py             # Model architecture + focal loss
│   ├── evaluate.py          # Hit@K, NDCG@K, Revenue@K, class collapse detection
│   └── table_prep/
│       ├── 01_train_prep.py     # SQL: create training base tables
│       ├── 02_infer_prep.py     # SQL: create inference base tables
│       ├── 03_channel_pref_prep.py
│       └── 04_final_base_prep.py
│
├── artifacts/               # (Generated) models, scalers, metrics
│   ├── atl_reco_taker.h5
│   ├── atl_reco_non_taker.h5
│   ├── atl_scaler_taker.pkl
│   ├── atl_scaler_non_taker.pkl
│   ├── label_encoder.pkl
│   ├── outlier_thresholds.json
│   └── eval_metrics.json
│
├── data/                    # (Generated) cached intermediate data
│   ├── processed_base.gz
│   ├── processed_base_infer.gz
│   ├── taker_predictions.gz
│   ├── nontaker_predictions.gz
│   └── base_pred.gz
│
└── notebooks/
    └── artifacts/
        └── atl_reco_dl.ipynb   # Exploratory notebook
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for training on ~20M samples)
- Oracle database access
- ~32GB RAM (for full inference pipeline)

### Python Dependencies

```
pandas
numpy
scikit-learn
tensorflow[and-cuda]
keras
tf2onnx
oracledb
SQLAlchemy
h5py
python-dotenv
```

## Installation

```bash
# Clone the repository
git clone https://github.com/sadmanaraf-gp/atl_recommendation.git
cd atl_recommendation

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
ORACLE_USER=your_username
ORACLE_PASSWORD=your_password
ORACLE_DSN=your_host:port/service_name
```

### Hyperparameters (`scripts/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_CLASSES` | 51 | Number of pack denominations |
| `EPOCHS` | 20 | Training epochs |
| `BATCH_SIZE` | 1024 | Training batch size |
| `VALIDATION_SPLIT` | 0.3 | Validation fraction |
| `FOCAL_LOSS_GAMMA` | 1.0 | Focal loss focusing parameter |
| `CLASS_WEIGHT_MODE` | `'sqrt'` | Weight dampening: `'sqrt'`, `'balanced'`, or `'none'` |
| `NONTAKER_HIT_QUANTILE` | 0.3 | Bottom N% of subscribers used as non-taker proxy |
| `TAKER_PROB_CUTOFF` | 0.05 | Min probability to include in taker recommendations |
| `NONTAKER_PROB_CUTOFF` | 0.03 | Min probability to include in non-taker recommendations |
| `PACK_NO` | 5 | Top-K recommendations per subscriber |

## Usage

### Train

```bash
python main.py train
```

This will:
1. Load/cache processed training data from Oracle
2. Train the Taker model (subscribers with pack_rank == 1, 70% sample)
3. Train the Non-Taker model (low-engagement proxy subscribers)
4. Evaluate both models and save metrics to `artifacts/eval_metrics.json`

### Predict

```bash
python main.py predict
```

This will:
1. Load/cache processed inference data
2. Run predictions on Taker and Non-Taker segments (split by `pack_flag`)
3. Extract top-K recommendations per subscriber
4. Export results to Oracle table (configured via `FINAL_TABLE` in config)

### Data Preparation (SQL)

Before training/inference, Oracle tables must be prepared:

```bash
python scripts/table_prep/01_train_prep.py    # Training tables
python scripts/table_prep/02_infer_prep.py    # Inference tables
```

These scripts create aggregated pack taker tables, base feature tables across 3 monthly partitions, and label tables.

### End-to-End Run

Complete steps to run the project from scratch:

```bash
# Step 1: Set up environment
git clone https://github.com/sadmanaraf-gp/atl_recommendation.git
cd atl_recommendation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Step 2: Configure Oracle credentials
# Create .env file with ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN

# Step 3: Prepare Oracle tables (run once per month or when data refreshes)
python scripts/table_prep/01_train_prep.py
python scripts/table_prep/02_infer_prep.py


# Step 4: Train models
python main.py train

# Step 5: Review evaluation metrics
cat artifacts/eval_metrics.json
# Check for class collapse warnings, dead classes, and Hit@5 > 80%

# Step 6: Run inference and export to Oracle
python main.py predict

# Step 7: Prepare the final table
python scripts/table_prep/03_channel_pref_prep.py
python scripts/table_prep/04_final_base_prep.py
```
**Monthly refresh cycle (Steps 3→6 only):**

```bash
# Delete cached data from previous run to force reload
Remove-Item data\processed_base.gz -ErrorAction SilentlyContinue
Remove-Item data\processed_base_infer.gz -ErrorAction SilentlyContinue
Remove-Item data\base_pred.gz -ErrorAction SilentlyContinue
Remove-Item data\taker_predictions.gz -ErrorAction SilentlyContinue
Remove-Item data\nontaker_predictions.gz -ErrorAction SilentlyContinue
Remove-Item data\base_taker_pred.gz -ErrorAction SilentlyContinue
Remove-Item data\base_nontaker_pred.gz -ErrorAction SilentlyContinue

# Re-run pipeline
python scripts/table_prep/01_train_prep.py
python scripts/table_prep/02_infer_prep.py
python scripts/table_prep/03_channel_pref_prep.py
python scripts/table_prep/04_final_base_prep.py
python main.py train
python main.py predict
```

**Quick re-inference (same model, new subscriber data):**

```bash
# Only delete inference cache
Remove-Item data\processed_base_infer.gz -ErrorAction SilentlyContinue
Remove-Item data\base_pred.gz -ErrorAction SilentlyContinue
Remove-Item data\taker_predictions.gz -ErrorAction SilentlyContinue
Remove-Item data\nontaker_predictions.gz -ErrorAction SilentlyContinue
Remove-Item data\base_taker_pred.gz -ErrorAction SilentlyContinue
Remove-Item data\base_nontaker_pred.gz -ErrorAction SilentlyContinue

# Update inference tables and run
python scripts/table_prep/02_infer_prep.py
python main.py predict
```

## Model Architecture

Tapering feedforward neural network with batch normalization:

```
Input (60-70 features)
  → Dense(2048, ReLU) → BatchNorm → Dropout(0.3)
  → Dense(1024, ReLU) → BatchNorm → Dropout(0.3)
  → Dense(512, ReLU)  → BatchNorm → Dropout(0.2)
  → Dense(256, ReLU)  → Dropout(0.2)
  → Dense(51, Softmax)
```

**Loss:** Focal loss (gamma=1.0, alpha=0.25) — addresses class imbalance by down-weighting easy/majority-class examples.

**Class Weights:** Balanced weights with sqrt dampening to avoid over-correction.


## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Hit@K | Proportion of samples where true label is in top-K predictions |
| NDCG@K | Normalized Discounted Cumulative Gain (rank-aware) |
| Revenue@K | Average pack denomination when prediction hits |
| Revenue Capture Rate | Revenue@K / potential revenue (ideally 90-100%) |
| Class Collapse Detection | Warns when any class captures >20% of predictions |
| Dead Classes | Packs that are never recommended |

## Artifacts

After training, the following files are generated in `artifacts/`:

| File | Description |
|------|-------------|
| `atl_reco_taker.h5` | Best Taker model weights (by val_loss) |
| `atl_reco_non_taker.h5` | Best Non-Taker model weights |
| `atl_scaler_taker.pkl` | StandardScaler for Taker features |
| `atl_scaler_non_taker.pkl` | StandardScaler for Non-Taker features |
| `label_encoder.pkl` | LabelEncoder mapping amount → label index |
| `outlier_thresholds.json` | Recharge/DSTR caps for inference clipping |
| `eval_metrics.json` | Full evaluation results for both models |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `load_weights()` shape mismatch | Model architecture changed — retrain before inference |
| `N_CLASSES mismatch` assertion | Pack universe changed — update `config.py` (N_CLASSES + class_names) |
| OOM during inference | Reduce `batch_size` in `run_inference()` |
| All predictions same class (collapse) | Increase `FOCAL_LOSS_GAMMA` or set `CLASS_WEIGHT_MODE='balanced'` |
| Revenue capture >100% (over-correction) | Reduce gamma or set `CLASS_WEIGHT_MODE='sqrt'` |
| Oracle export slow | Increase `EXPORT_BATCH_SIZE` in config |