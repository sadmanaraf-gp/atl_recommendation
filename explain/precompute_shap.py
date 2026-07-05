"""Offline SHAP precompute job for the explainability dashboard.

Draws a stratified sample of subscribers per model, computes SHAP values for
each subscriber's top-ranked (recommended) pack, and caches the results to
artifacts/shap_cache/. The dashboard's Global Drivers and Feature Dependence
views read these caches and never call the model live.

Run from the project root:
    python -m explain.precompute_shap [--sample 8000]

Inputs (produced by the existing pipeline):
    data/processed_base_infer.gz  - preprocessed inference features + msisdn + pack_flag
    data/base_pred.gz             - top-5 predictions (msisdn, deno, prob, rank, taker_flag)

Outputs:
    artifacts/shap_cache/global_taker.parquet
    artifacts/shap_cache/global_non_taker.parquet
    artifacts/shap_cache/sample_features_<kind>.parquet  (raw model inputs for offline lookup)
    artifacts/shap_cache/meta.json
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import gzip
import json
import pickle
import argparse

import numpy as np
import pandas as pd

from explain import shap_utils as su

PROCESSED_INFER = os.path.join(su.ROOT, "data", "processed_base_infer.gz")
BASE_PRED = os.path.join(su.ROOT, "data", "base_pred.gz")

# pack_flag value in the inference frame -> model_kind
FLAG_TO_KIND = {"TAKER": "taker", "NON_TAKER": "non_taker"}


def _load_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _stratified_sample(df, strat_col, n, seed=42):
    """Sample ~n rows, spread across the values of strat_col so rare packs are
    represented. Falls back to a plain sample if the column is absent.

    Samples *indices* per group and selects with .loc so every column is
    preserved (pandas 3.x drops grouping columns inside groupby.apply)."""
    if n >= len(df) or strat_col not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=seed)
    frac = n / len(df)
    picked = df.groupby(strat_col, group_keys=False).apply(
        lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed)
    ).index
    out = df.loc[picked]
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out


def precompute_for_kind(model_kind, infer_df, top_pack, sample_size):
    print(f"\n=== {model_kind} ===")
    features = su.features_for(model_kind)

    # Attach each subscriber's recommended (rank-1) pack, then sample.
    df = infer_df.merge(top_pack, on="msisdn", how="inner")
    print(f"  subscribers with a rank-1 prediction: {len(df):,}")
    if df.empty:
        print("  nothing to do for this kind, skipping.")
        return None

    sample = _stratified_sample(df, "deno", sample_size)
    print(f"  sampled: {len(sample):,}")

    art = su.load_artifacts(model_kind)
    model, scaler = art["model"], art["scaler"]

    # Align to the model's feature list; fill any absent one-hot columns with 0.
    missing = [c for c in features if c not in sample.columns]
    if missing:
        print(f"  note: {len(missing)} feature columns absent in cache, filled with 0: {missing[:6]}...")
    feat_frame = sample.reindex(columns=features, fill_value=0)
    X = feat_frame.to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)

    explainer = su.make_explainer(model, X_scaled)
    print("  computing SHAP for top-1 pack per subscriber...")
    shap_mat, class_idx = su.shap_top1(explainer, X_scaled)

    # Persist: one row per subscriber, columns = shap_<feat>, val_<feat>.
    out = pd.DataFrame({"msisdn": sample["msisdn"].to_numpy(),
                        "deno": sample["deno"].to_numpy()})
    shap_cols = pd.DataFrame(shap_mat, columns=[f"shap_{c}" for c in features],
                             index=out.index)
    val_cols = pd.DataFrame(X, columns=[f"val_{c}" for c in features], index=out.index)
    out = pd.concat([out, shap_cols, val_cols], axis=1)

    os.makedirs(su.SHAP_CACHE, exist_ok=True)
    global_path = os.path.join(su.SHAP_CACHE, f"global_{model_kind}.parquet")
    out.to_parquet(global_path, index=False)
    print(f"  wrote {global_path} ({len(out):,} rows)")

    # Raw model-input features for the sampled subscribers, so the dashboard can
    # do offline (no-Oracle) per-subscriber SHAP on any of their top-5 packs.
    feat_store = pd.concat([sample[["msisdn"]].reset_index(drop=True),
                            feat_frame.reset_index(drop=True)], axis=1)
    feat_path = os.path.join(su.SHAP_CACHE, f"sample_features_{model_kind}.parquet")
    feat_store.to_parquet(feat_path, index=False)
    print(f"  wrote {feat_path}")

    # Quick sanity: top global drivers by mean |SHAP|.
    mean_abs = np.abs(shap_mat).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:8]
    print("  top global drivers (mean |SHAP|):")
    for i in order:
        print(f"    {features[i]:<28} {mean_abs[i]:.5f}")

    return {"model_kind": model_kind, "n": int(len(out)), "n_features": len(features)}


def main():
    parser = argparse.ArgumentParser(description="Precompute SHAP caches for the dashboard.")
    parser.add_argument("--sample", type=int, default=8000,
                        help="Subscribers to sample per model (default 8000).")
    args = parser.parse_args()

    print("Loading inference features and predictions from cache...")
    infer = _load_gz(PROCESSED_INFER)
    infer.columns = infer.columns.str.lower()
    pred = _load_gz(BASE_PRED)

    # Rank-1 (recommended) pack per subscriber.
    top_pack = (pred[pred["rank"] == 1][["msisdn", "deno"]]
                .drop_duplicates("msisdn"))

    meta = {"sample_size": args.sample, "models": []}
    for flag, kind in FLAG_TO_KIND.items():
        sub = infer[infer["pack_flag"] == flag]
        info = precompute_for_kind(kind, sub, top_pack, args.sample)
        if info:
            meta["models"].append(info)

    os.makedirs(su.SHAP_CACHE, exist_ok=True)
    with open(os.path.join(su.SHAP_CACHE, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Caches in {su.SHAP_CACHE}")


if __name__ == "__main__":
    main()
