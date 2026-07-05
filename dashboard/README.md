# Explainability Dashboard

SHAP-based transparency for the ATL pack-recommendation models (taker / non-taker
Keras nets, 58 pack denominations). Answers: *what drives recommendations overall*,
*why did this subscriber get these packs*, and *how does a feature move pack probability*.

## Setup

```bash
pip install -r requirements.txt          # adds shap, streamlit, matplotlib, plotly, pyarrow
```

Requires a GPU/CPU environment with TensorFlow installed and the trained artifacts
in `artifacts/` (`atl_reco_*.h5`, `atl_scaler_*.pkl`, `label_encoder.pkl`) plus the
cached inference data in `data/` (`processed_base_infer.gz`, `base_pred.gz`).

## 1. Precompute the global SHAP caches (offline, run once per model refresh)

```bash
python -m explain.precompute_shap --sample 8000
```

Writes `artifacts/shap_cache/global_{taker,non_taker}.parquet`,
`sample_features_{taker,non_taker}.parquet`, and `meta.json`. The job prints the
top global drivers per model as a sanity check.

## 2. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Views:
- **Global Drivers** — mean-|SHAP| importance + beeswarm, per model, optionally per pack.
- **Subscriber Lookup** — pick/enter an MSISDN, see top-5 packs and a per-subscriber
  SHAP explanation (waterfall + plain-language narrative). Uses the offline sample
  cache by default; set `ORACLE_USER/PASSWORD/DSN` to fetch any live subscriber.
- **Feature Dependence** — how one feature's value maps to its SHAP contribution.
- **Model Health** — hit@k / NDCG / revenue-capture and class-collapse warnings
  from `artifacts/eval_metrics.json`.

The explainer is `shap.GradientExplainer` (robust to the models' BatchNorm layers).
```
