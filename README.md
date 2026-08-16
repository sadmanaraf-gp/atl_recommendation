# ATL Pack Recommendation Pipeline

A production ML pipeline that recommends ATL packs to Grameenphone
subscribers. It builds its feature base directly in Oracle, trains two deep-learning models
(Taker / Non-Taker), scores the full subscriber base (~80M MSISDNs), and writes a
campaign-ready recommendation table back to Oracle — followed by an in-market back-test and
a SHAP + LLM explainability dashboard.

Everything is driven by one orchestrator: **`python main.py`**.

## Table of Contents

- [Overview](#overview)
- [Architecture and Data Flow](#architecture-and-data-flow)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Manual / Out-of-Band Steps](#manual--out-of-band-steps)
- [Explainability: SHAP, LLM, Dashboard](#explainability-shap-llm-dashboard)
- [Monthly Refresh Runbook](#monthly-refresh-runbook)
- [Evaluation](#evaluation)
- [Artifacts and Caches](#artifacts-and-caches)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)

## Overview

1. **Builds** the training and inference feature bases in Oracle via parallel DDL
   (3 monthly partitions of usage, revenue, recharge and pack-purchase history).
2. **Preprocesses** in pandas: one-hot encoding, outlier caps, month-over-month diffs,
   trend/volatility features, and label encoding of pack denominations.
3. **Trains** two multi-class models over 56 pack denominations:
   - **Taker** — subscribers with pack-purchase history (uses `amount1..8` / `srvc1..8`).
   - **Non-Taker** — low-engagement subscribers (proxy: bottom 30% by pack hit count).
4. **Evaluates** offline with Hit@K, NDCG@K, revenue capture and class-collapse detection.
5. **Scores** the full base, keeping the top 5 packs per subscriber, and exports to Oracle.
6. **Post-processes** into the campaign table: 5% control group, upsell vs. probability
   segmentation, and preferred contact channel (MyGP / USSD / MFS / Trigger).
7. **Back-tests** last month's recommendations against actual purchases.
8. **Explains** the models with SHAP, optionally narrated by a locally served LLM.

## Architecture and Data Flow

```text
Oracle sources
  GPBI_MARKETING.GPBI_SUBSCRIPTION_REV_DAILY
  GPBI_MARKETING.GPBI_MSTR_DETAIL_INFO        (partitioned PART_DDMMYYYY: T, T-1, T-2)
  GPBI_MARKETING.GPBI_CUSTOMER_DIM
  GPBI_MARKETING.GPBI_MSISDN_EXCLUSION_LIST   (SKITTO exclusions)
  OCDM_SYS.DWB_PRPD_RCHRG                     (recharges)
  TBL_ATL_MAPPING                             (pack -> AMOUNT / SRVC / IDENTIFIER)
        |
  [1] 01_train_prep      -> SA_TBL_AGG_* / SA_TBL_PACK_TAKERS_<LABEL_MONTH>
                         -> SA_TBL_ATL_BASE_01 / _02 / _03
        |
  [2] train              -> data/base_train.gz, data/processed_base.gz
                         -> artifacts/atl_reco_{taker,non_taker}.h5
                            artifacts/atl_scaler_{taker,non_taker}.pkl
                            artifacts/label_encoder.pkl
                            artifacts/outlier_thresholds.json
                            artifacts/eval_metrics.json
        |
  [3] 02_infer_prep      -> SA_TBL_ATL_BASE_INFER_01 / _02 / _03  (PACK_FLAG = TAKER|NON_TAKER)
        |
  [4] predict            -> data/base_infer.gz, data/processed_base_infer.gz, data/base_pred.gz
                         -> SA_TBL_ATL_PRED_<YYYYMM>        (top-5 packs per MSISDN)
        |
  [5] 03_channel_pref    -> TBL_CHANNEL_PREFERENCE_01 -> TBL_CHANNEL_PREFERENCE
                         -> TBL_CHANNEL_PREFERENCE_FINAL
        |
  [6] 04_final_base      -> SA_TBL_ATL_PRED_<PM>_UNIQUE -> TBL_ATL_PREDICTION_<PM>_1
                         -> TMP_RECO_SEGM_1 / _2 -> _UPSELL / _PROB -> _2 -> _3
                         -> TBL_ATL_PREDICTION_<PM>        <== campaign-ready output
        |
  [7] post_evaluation    -> SA_TBL_EVAL_*_<PM>, SA_TBL_EVAL_RESULTS_<PM>  + printed metrics

  (manual) 05_table_prep_mygp -> TBL_ATL_PREDICTION_<PM>_MYGP
  (manual) explain.precompute_shap -> artifacts/shap_cache/*  -> dashboard/app.py
```

`<PM>` / `<YYYYMM>` is the current calendar month; the scripts derive their month keys from
`datetime.now()`, so the run date determines which tables are read and written.

## Project Structure

```text
├── main.py                       # Pipeline orchestrator (all / clean / single step)
├── train.py                      # Trains both models, writes artifacts + eval metrics
├── predict.py                    # Batch inference + top-K extraction + Oracle export
├── requirements.txt
├── .env                          # Credentials and LLM settings (gitignored)
│
├── scripts/
│   ├── config.py                 # Feature lists, hyperparameters, Oracle table names
│   ├── data_processing.py        # load_and_process_data(), preprocess_data()
│   ├── db_utils.py               # read_oracle_data(), export_to_oracle()
│   ├── model.py                  # build_model(), focal_loss()
│   ├── evaluate.py               # Hit@K, NDCG@K, Revenue@K, class-collapse checks
│   ├── serve_llm.sh              # Launches the local vLLM OpenAI-compatible server
│   └── table_prep/
│       ├── 01_train_prep.py           # [step 1] training base tables
│       ├── 02_infer_prep.py           # [step 3] inference base tables
│       ├── 03_channel_pref_prep.py    # [step 5] channel preference
│       ├── 04_final_base_prep.py      # [step 6] campaign table
│       ├── post_evaluation.py         # [step 7] in-market back-test
│       ├── post_evaluation_dynamic.py # manual: parameterised back-test (argparse)
│       └── 05_table_prep_mygp.py      # manual: MyGP channel base
│
├── explain/
│   ├── precompute_shap.py        # Offline SHAP caches for the dashboard
│   ├── shap_utils.py             # Explainer construction, feature naming/ordering
│   └── llm_explain.py            # Optional LLM narration of SHAP drivers
│
├── dashboard/
│   └── app.py                    # Streamlit explainability dashboard (4 views)
│
├── artifacts/                    # (generated, gitignored) models, scalers, metrics, shap_cache/
├── data/                         # (generated, gitignored) cached dataframes, ~35 GB at full scale
└── notebooks/
    └── artifacts/atl_reco_dl.ipynb   # Original exploratory notebook
```

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | 3.11 used in the current environment |
| CUDA-capable GPU | Required in practice for training (~20M samples) and SHAP |
| ~32 GB RAM | Full-base inference loads multi-GB frames |
| ~50 GB free disk | `data/` reaches ~35 GB (`processed_base_infer.gz` alone is ~14 GB) |
| Oracle access | `oracledb` runs in **thin mode** — no Oracle client install needed |

Read/write privileges are needed on the working schema: the pipeline creates and drops
`SA_TBL_*`, `TBL_ATL_PREDICTION_*`, `TBL_CHANNEL_PREFERENCE*` and `TMP_RECO_SEGM_*` tables.

## Installation

```bash
git clone https://github.com/sadmanaraf-gp/atl_recommendation.git
cd atl_recommendation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Optional: the LLM explanation server

vLLM is installed in a **separate** virtualenv to avoid torch/CUDA conflicts with
`tensorflow[and-cuda]`:

```bash
python -m venv .venv-vllm
.venv-vllm/bin/pip install "vllm==0.6.*" "transformers==4.47.1"
```

`transformers` must stay pinned to 4.47.x — vllm 0.6.6 breaks on newer releases
(`Qwen2Tokenizer has no attribute all_special_tokens_extended`).

## Configuration

### `.env` (project root, gitignored)

```env
# Oracle
ORACLE_USER=your_username
ORACLE_PASSWORD=your_password
ORACLE_DSN=host:port/service_name
ORACLE_HOST=host
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=service_name
ORACLE_ENCODING=UTF-8

# Local vLLM OpenAI-compatible server (optional).
# Leave LLM_BASE_URL empty to disable the LLM layer — the dashboard then falls back
# to the mechanical bullet-point narrative.
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=qwen2.5-7b
LLM_API_KEY=EMPTY
LLM_TIMEOUT=30
```

`ORACLE_USER` / `ORACLE_PASSWORD` / `ORACLE_DSN` are the three that every step requires;
the host/port/service/encoding keys are used by the dashboard's live-lookup path.

### Hyperparameters (`scripts/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `N_CLASSES` | 56 | Pack denominations (must equal `label_encoder.pkl` size) |
| `NODES` | 2048 | Width of the first dense layer |
| `DROPOUT_RATE` | 0.3 | Dropout on the first two blocks |
| `EPOCHS` | 20 | Training epochs |
| `BATCH_SIZE` | 1024 | Training batch size |
| `VALIDATION_SPLIT` | 0.3 | Validation fraction |
| `USE_FOCAL_LOSS` | `True` | Focal loss vs. label-smoothed cross-entropy |
| `FOCAL_LOSS_GAMMA` / `_ALPHA` | 1.0 / 0.25 | Focal loss parameters |
| `CLASS_WEIGHT_MODE` | `'sqrt'` | `'sqrt'`, `'balanced'` or `'none'` |
| `NONTAKER_HIT_QUANTILE` | 0.3 | Bottom N% by hit count = non-taker proxy |
| `TAKER_PROB_CUTOFF` | 0.05 | Min probability kept for takers |
| `NONTAKER_PROB_CUTOFF` | 0.03 | Min probability kept for non-takers |
| `PACK_NO` | 5 | Top-K recommendations per subscriber |
| `EXPORT_BATCH_SIZE` | 1559752 | Rows per Oracle `executemany` batch |

### Table names (`scripts/config.py`)

`TABLE_BASE_01/02/03`, `TABLE_INFER_01/02/03`, and:

```python
TABLE_PRED_PREFIX = "SA_TBL_ATL_PRED_"
FINAL_TABLE = TABLE_PRED_PREFIX + datetime.now().strftime("%Y%m")
```

`FINAL_TABLE` is derived from the **current month**, so a run in August 2026 writes
`SA_TBL_ATL_PRED_202608`. To re-run for a past month, override `FINAL_TABLE` explicitly.

## Running the Pipeline

```bash
python main.py                  # 'all' — the 7 steps below, in order
python main.py all --clean      # wipe artifacts/ and data/ first, then full run
python main.py <step>           # run a single step (resume after a failure)
python main.py clean            # wipe artifacts/ and data/ only
```

`main.py` runs the steps sequentially and **aborts on the first non-zero exit code**, so a
failed step stops the run rather than corrupting downstream tables.

> **Warning:** `--clean` and `clean` delete *every* file in `artifacts/` and `data/` —
> including the trained models and ~35 GB of cached frames. Only use them when you intend a
> full retrain from Oracle.

| # | Step name | Reads | Writes | Runtime |
|---|---|---|---|---|
| 1 | `01_train_prep` | Oracle sources (T-2 features, T-1 labels) | `SA_TBL_ATL_BASE_01/02/03` | hours |
| 2 | `train` | base tables (or `data/processed_base.gz`) | both `.h5` models, scalers, label encoder, `eval_metrics.json` | hours (GPU) |
| 3 | `02_infer_prep` | Oracle sources (current month) | `SA_TBL_ATL_BASE_INFER_01/02/03` | hours |
| 4 | `predict` | INFER tables (or `data/processed_base_infer.gz`) + artifacts | `data/base_pred.gz`, `SA_TBL_ATL_PRED_<YYYYMM>` | hours |
| 5 | `03_channel_pref` | recharge/purchase history (last 3 months) | `TBL_CHANNEL_PREFERENCE_FINAL` | ~1 h |
| 6 | `04_final_base` | `SA_TBL_ATL_PRED_<PM>`, channel prefs, `TBL_ATL_MAPPING` | `TBL_ATL_PREDICTION_<PM>` | ~1–2 h |
| 7 | `post_evaluation` | `SA_TBL_ATL_PRED_<PM>` vs. actual purchases MTD | `SA_TBL_EVAL_*_<PM>` + printed metrics | ~30 m |

Runtimes are order-of-magnitude for the full ~80M base and vary with EDW load.

### Resuming safely

Every step is designed to be re-runnable:

- **Table prep** — `DROP TABLE ... PURGE` tolerates `ORA-00942` (table absent), and
  `01_/02_train/infer_prep` skip a `CREATE` when the table already exists (`table_exists()`).
- **Oracle export** — `export_to_oracle()` creates the table if missing, otherwise
  `TRUNCATE`s it, so a repeat export replaces rather than duplicates rows.
- **Dataframe caches** — `train.py` and `predict.py` reuse `data/processed_base.gz` and
  `data/processed_base_infer.gz` when they exist, skipping the multi-hour Oracle read.
  Delete just those files (or use `--clean`) to force a reload.
- **`04_final_base_prep.py`** additionally uses TCP keepalive (`expire_time=1`), retries
  connection loss up to 3 times, and drops half-built tables before retrying.

## Manual / Out-of-Band Steps

These are intentionally **not** part of `main.py all`:

```bash
# MyGP channel base — run after step 6 (04_final_base)
python scripts/table_prep/05_table_prep_mygp.py
# -> TBL_ATL_PREDICTION_<PM>_{RECO_BASE,NON_RECO_BASE,NON_RECO,MYGP}
#    Includes built-in sanity tests (deno spread, flag breakdown, NULL and overlap checks).



# Back-test an arbitrary month / window
python scripts/table_prep/post_evaluation_dynamic.py \
    --pred-month 202607 \
    --eval-start 2026-07-01 --eval-end 2026-07-31 \
    [--mapping-day-key 31-JUL-26] [--rebuild]
```

Dates accept `YYYY-MM-DD` or `DD-MON-YY`. `--rebuild` drops and recreates the evaluation
tables; without it, existing tables are reused and only the metrics are reported.

## Explainability: SHAP, LLM, Dashboard

Requires the trained artifacts in `artifacts/` and the inference caches in `data/`
(`processed_base_infer.gz`, `base_pred.gz`) — i.e. run steps 2 and 4 first.

```bash
# 1. (Optional) start the LLM server in a separate shell
bash scripts/serve_llm.sh
curl http://127.0.0.1:8000/v1/models          # smoke test

# 2. Precompute the SHAP caches (once per model refresh)
python -m explain.precompute_shap --sample 8000 [--no-llm] [--llm-workers 8]

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

`precompute_shap` writes `artifacts/shap_cache/`:
`global_{taker,non_taker}.parquet`, `sample_features_{taker,non_taker}.parquet`, `meta.json`.
When `LLM_BASE_URL` is set it also adds an `explanation` column with LLM-written narratives;
`--no-llm` skips that. Any LLM failure degrades gracefully to the mechanical bullet text.

Dashboard views:

- **Global Drivers** — mean-|SHAP| importance and beeswarm, per model, optionally per pack.
- **Subscriber Lookup** — top-5 packs plus a per-subscriber waterfall and narrative. Uses the
  offline sample cache by default; with `ORACLE_*` set it can fetch any live MSISDN.
- **Feature Dependence** — how a feature's value maps to its SHAP contribution.
- **Model Health** — Hit@K / NDCG / revenue capture and collapse warnings from
  `artifacts/eval_metrics.json`.

The explainer is `shap.GradientExplainer` (robust to the models' BatchNorm layers).

> **Feature order:** `explain/shap_utils.py` derives the feature list from the fitted scaler
> (`scaler.feature_names_in_`), *not* from `scripts/config.py`. Always treat the scaler as the
> source of truth when the two disagree.

## Monthly Refresh Runbook

The standard monthly cycle, run on or shortly after the 1st:

```bash
# 0. Pre-flight
#    - .env credentials valid
#    - disk has ~50 GB free
#    - scripts/table_prep/04_final_base_prep.py has ALL its SQL blocks active (see note)

# 1. Full refresh, retraining from scratch
python main.py all --clean

# 2. MyGP channel base
python scripts/table_prep/05_table_prep_mygp.py

# 3. Refresh explainability
python -m explain.precompute_shap --sample 8000
```

**Reusing last month's models** (skip retraining — the base tables and models stay put):

```bash
rm -f data/processed_base_infer.gz data/base_infer.gz data/base_pred.gz
python main.py 02_infer_prep
python main.py predict
python main.py 03_channel_pref
python main.py 04_final_base
python main.py post_evaluation
```

> **Pre-flight note on `04_final_base_prep.py`:** its `STATEMENTS` list is sometimes partially
> commented out during debugging to re-run only the tail of the chain. Before a monthly run,
> confirm the `_UNIQUE`, `_1`, `TMP_RECO_SEGM_*`, `_UPSELL` and `_PROB` blocks are all active —
> otherwise the step either fails on a missing table or silently rebuilds the final table from
> last month's intermediates.

## Evaluation

Two independent layers:

**1. Offline (`train.py` → `artifacts/eval_metrics.json`)** — held-out test split per model.

| Metric | Description |
|---|---|
| Hit@K | Share of samples where the true pack is in the top K |
| NDCG@K | Rank-aware gain |
| Revenue@K | Average denomination when the prediction hits |
| Revenue capture rate | Revenue@K / potential revenue (healthy: 0.9–1.0) |
| Class collapse | Warns when one class takes >20% of predictions |
| Dead classes | Packs never recommended |

Current run: Taker Hit@1 0.477 / Hit@5 0.849, capture 0.976 (6.26M test rows);
Non-Taker Hit@1 0.444 / Hit@5 0.793, capture 1.048 (2.84M rows); no collapse or dead classes.

**2. In-market (`post_evaluation.py`)** — last month's exported recommendations against actual
purchases, from the 1st of the month to today. Reports `total_predictions`,
`unique_subscribers_predicted`, `actual_takers_this_month`, `hit_at_1/3/5`, hit@1 split by
taker/non-taker, and the top-15 predicted packs. `evaluation.log` holds a worked example
(01-MAY-26 → 20-MAY-26: 336M predictions over 81.6M subscribers, hit@1 68.3, hit@5 88.5).

The in-market numbers run far above the offline ones because a subscriber counts as a hit on
*any* qualifying purchase in the window, not on a single held-out label.

## Artifacts and Caches

`artifacts/` (gitignored):

| File | Description |
|---|---|
| `atl_reco_taker.h5` | Best Taker weights (by `val_loss`) |
| `atl_reco_non_taker.h5` | Best Non-Taker weights |
| `atl_scaler_taker.pkl` | `StandardScaler` for taker features — also the feature-order source of truth |
| `atl_scaler_non_taker.pkl` | `StandardScaler` for non-taker features |
| `label_encoder.pkl` | `LabelEncoder`: amount → class index (56 classes) |
| `outlier_thresholds.json` | Recharge / DSTR caps used to clip at inference |
| `eval_metrics.json` | Offline metrics for both models |
| `shap_cache/` | Dashboard caches from `explain.precompute_shap` |

`data/` (gitignored, ~35 GB at full scale):

| File | Contents |
|---|---|
| `base_train.gz` | Raw 3-table training merge |
| `processed_base.gz` | Post-preprocessing training frame |
| `base_infer.gz` | Raw 3-table inference merge |
| `processed_base_infer.gz` | Post-preprocessing inference frame |
| `base_pred.gz` | Top-5 predictions before Oracle export |

### Model architecture

```text
Input (features from the fitted scaler)
  -> Dense(2048, ReLU) -> BatchNorm -> Dropout(0.3)
  -> Dense(1024, ReLU) -> BatchNorm -> Dropout(0.3)
  -> Dense(512,  ReLU) -> BatchNorm -> Dropout(0.2)
  -> Dense(256,  ReLU) -> Dropout(0.2)
  -> Dense(56, Softmax)
```

Loss: focal loss (gamma 1.0, alpha 0.25) to counter class imbalance.
Class weights: balanced with sqrt dampening.
Callbacks: `ModelCheckpoint(save_best_only)`, `ReduceLROnPlateau(0.5, patience=3)`,
`EarlyStopping(patience=5, restore_best_weights=True)`.

## Troubleshooting

| Issue | Cause / fix |
|---|---|
| `load_weights()` shape mismatch | Feature set or `N_CLASSES` changed — retrain before inference |
| `N_CLASSES mismatch` assertion in `train.py` | Pack universe changed; update `N_CLASSES` in `scripts/config.py` |
| `FileNotFoundError: artifacts/label_encoder.pkl` on `import predict` | `predict.py` loads the encoder at import time — run `train` first |
| Feature list disagrees with the model | Use `scaler.feature_names_in_`, not `config.py` — see `explain/shap_utils.py` |
| OOM during inference | Lower the `batch_size` argument of `run_inference_top_k()` (default 2,000,000) |
| Predictions collapse to one class | Raise `FOCAL_LOSS_GAMMA` or set `CLASS_WEIGHT_MODE='balanced'` |
| Revenue capture > 1.0 (over-correction) | Lower gamma or set `CLASS_WEIGHT_MODE='sqrt'` |
| Oracle export slow | Raise `EXPORT_BATCH_SIZE` in `scripts/config.py` |
| `ORA-03113` / connection lost mid-DDL | Long EDW statements drop idle sessions; `04_final_base_prep.py` retries 3× and cleans partial tables — re-run the step |
| `ORA-00942` on a `CREATE ... AS SELECT` | An upstream table is missing — re-run the earlier step |
| `04_final_base` fails on a missing `_UPSELL` / `_PROB` | Its SQL blocks are commented out; re-enable them (see the runbook note) |
| Dashboard: "no SHAP cache" | Run `python -m explain.precompute_shap --sample 8000` |
| Dashboard shows bullets, not narratives | `LLM_BASE_URL` unset or the vLLM server is down |
| Subscriber lookup is very slow | `SA_TBL_ATL_BASE_INFER_*` have no MSISDN index; the dashboard already queries the three tables in parallel with a `PARALLEL(8)` hint |
| `ModuleNotFoundError: dateutil` | `pip install python-dateutil` (used by the table-prep scripts) |
| vLLM: `Qwen2Tokenizer has no attribute all_special_tokens_extended` | `transformers` is too new — pin `4.47.1` in `.venv-vllm` |

## Security Notes

- `.env` is gitignored; never commit credentials.
- **`notebooks/artifacts/atl_reco_dl.ipynb` contains plaintext EDW credentials in cell 5.**
  Those credentials should be rotated and stripped from the notebook (and ideally from git
  history). The notebook is exploratory only and is not part of the pipeline.
