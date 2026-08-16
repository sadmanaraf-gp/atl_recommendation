# Explainability Dashboard

SHAP-based transparency for the ATL pack-recommendation models (taker / non-taker Keras nets,
56 pack denominations). Answers: *what drives recommendations overall*, *why did this
subscriber get these packs*, and *how does a feature move pack probability*.

Setup, the SHAP precompute step, the optional LLM narration layer, and the list of views are
documented in the root [README](../README.md#explainability-shap-llm-dashboard).

Quick start (after `train` and `predict` have run):

```bash
python -m explain.precompute_shap --sample 8000
streamlit run dashboard/app.py
```
