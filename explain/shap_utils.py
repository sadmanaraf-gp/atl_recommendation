"""Shared explainability core for the ATL pack-recommendation models.
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*np.object.*FutureWarning.*",
    category=FutureWarning,
)
import os
import pickle

import numpy as np
import pandas as pd

from scripts.model import build_model
from scripts.data_processing import preprocess_data

# Resolve paths relative to the project root (parent of this file's dir) so the
# helpers work no matter what the working directory is.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
SHAP_CACHE = os.path.join(ARTIFACTS, "shap_cache")

# model_kind -> (weights file, scaler file).
# NOTE: the feature list is taken from the fitted scaler's `feature_names_in_`,
# NOT from scripts.config. The config lists (TAKER_FEATURES/NONTAKER_FEATURES)
# have drifted past the deployed artifacts; the scaler is the single source of
# truth that is guaranteed to match both the saved model's input dim and the
# scaling, in the exact column order the model expects.
_KIND = {
    "taker": ("atl_reco_taker.h5", "atl_scaler_taker.pkl"),
    "non_taker": ("atl_reco_non_taker.h5", "atl_scaler_non_taker.pkl"),
}

# Business-friendly aliases for the model's raw feature names. Feature meanings
# confirmed with the model owner (2026-07): dstr = total spend (all services);
# *_trig = trigger-based purchase revenue; *_rg_days = revenue-generating days;
# amount1..8 / srvc1..8 = the price and service type of the customer's top-8
# most-purchased packs (ranked by purchase count).
#
# Base (current-month) names. Lag/change suffixes and one-hot groups are handled
# programmatically in friendly_name(), so only the roots live here.
_BASE = {
    "total_dstr": "Total spend",
    "voicerev_total": "Voice revenue",
    "datarev_total": "Data revenue",
    "mixed_bundle_rev": "Bundle revenue",
    "voicerev_trig": "Voice revenue (trigger)",
    "datarev_trig": "Data revenue (trigger)",
    "vol_mb": "Data usage (MB)",
    "mo_mou": "Voice usage (minutes)",
    "recharge_cnt": "Recharge count",
    "recharge_amount": "Total recharge amount",
    "recharge_max": "Largest recharge",
    "data_rg_days": "Data active (paid) days",
    "voice_rg_days": "Voice active (paid) days",
    "days_since_last_rcrg": "Days since last recharge",
    "mygp_m1_act": "MyGP active last month",
    "mygp_m2_act": "MyGP active 2 months ago",
    "mygp_active_days": "MyGP active days",
    "mygp_pack_rev": "MyGP pack revenue",
    "mygp_pack_hits": "MyGP packs bought",
    "rchg_amt_mygp": "MyGP recharge amount",
    "rchg_amt_max_mygp": "Largest MyGP recharge",
    "rchg_amt_digtal": "Digital recharge amount",
    "rchg_amt_max_digital": "Largest digital recharge",
    # Engineered features (not used by the deployed 78/62-feature models, but
    # kept so labels stay friendly if the feature set changes).
    "data_rev_share": "Share of revenue from data",
    "voice_rev_share": "Share of revenue from voice",
    "bundle_rev_share": "Share of revenue from bundles",
    "avg_recharge_value": "Average recharge value",
    "rfm_score": "Overall engagement (RFM)",
    "digital_recharge_ratio": "Digital-recharge ratio",
    "mygp_recharge_ratio": "MyGP-recharge ratio",
    "mygp_active_ratio": "MyGP active ratio",
    "vol_trend": "Data-usage trend",
    "mou_trend": "Voice-usage trend",
    "datarev_trend": "Data-revenue trend",
    "smartphone_data_user": "Smartphone data user",
    "is_declining_data": "Declining-data flag",
}

# Roots for month-over-month change features (dstr/mou/vol only).
_CHANGE_ROOT = {"dstr": "Total spend", "mou": "Voice usage", "vol": "Data usage"}

_RCHG_CHNL = {
    "01. freq_retail_only": "retail only",
    "02. freq_retail_dominant": "mostly retail",
    "03. freq_mixed": "mixed",
    "04. freq_digital_dominant": "mostly digital",
    "05. freq_digital_only": "digital only",
    "unknown": "unknown",
}


def friendly_name(feature):
    """Business-friendly label for a raw feature name.

    Handles the raw roots plus programmatic suffixes/groups:
      *_1 / *_2            -> "(last month)" / "(2 months ago)"
      *_change_01 / _12    -> month-over-month change labels
      amount{n} / srvc{n}  -> "Top #n purchased deno" / "Top #n purchased deno"
      circle_* / rchg_chnl_* -> "Region: ..." / "Recharge channel: ..."
    """
    f = feature.lower()

    # Month-over-month change features, e.g. dstr_change_01 / vol_change_12.
    if "_change_" in f:
        root, _, period = f.partition("_change_")
        base = _CHANGE_ROOT.get(root, root.replace("_", " "))
        if period == "01":
            return f"{base} change (vs last month)"
        if period == "12":
            return f"{base} change (last vs 2 months ago)"
        return f"{base} change"

    # Lagged versions of a base metric.
    for suffix, label in (("_1", " (last month)"), ("_2", " (2 months ago)")):
        if f.endswith(suffix):
            root = f[: -len(suffix)]
            if root in _BASE:
                return _BASE[root] + label

    # Customer's top-N most-purchased packs.
    if f.startswith("amount") and f[6:].isdigit():
        return f"Top #{f[6:]} pack"
    if f.startswith("srvc") and f[4:].isdigit():
        return f"Top #{f[4:]} service"

    # Geographic region (one-hot).
    if f.startswith("circle_"):
        region = f[len("circle_"):]
        return "Region: " + ("unknown" if region == "unknown" else region.title())

    # Recharge-channel behavior (one-hot). Original names keep a space/period.
    if f.startswith("rchg_chnl_"):
        key = feature[len("rchg_chnl_"):]
        return "Recharge channel: " + _RCHG_CHNL.get(key, key.replace("_", " "))

    if f in _BASE:
        return _BASE[f]
    return feature.replace("_", " ")


# Unit hints for value formatting, keyed by base (current-month) feature name.
# Bangladesh Taka for money; others as noted. Lags (_1/_2) inherit their base.
_UNITS = {
    "total_dstr": "TK", "voicerev_total": "TK", "datarev_total": "TK",
    "mixed_bundle_rev": "TK", "voicerev_trig": "TK", "datarev_trig": "TK",
    "recharge_amount": "TK", "recharge_max": "TK", "mygp_pack_rev": "TK",
    "rchg_amt_mygp": "TK", "rchg_amt_max_mygp": "TK",
    "rchg_amt_digtal": "TK", "rchg_amt_max_digital": "TK",
    "vol_mb": "MB", "mo_mou": "min",
    "data_rg_days": "days", "voice_rg_days": "days",
    "mygp_active_days": "days", "days_since_last_rcrg": "days",
}

# srvc{n} encodes the service type of the customer's n-th most-bought pack.
# Encoding (from the SQL mapping): DATA=1, VOICE=2, MIXED=3, RC (recharge)=4.
_SRVC_CODES = {1: "Data", 2: "Voice", 3: "Mixed", 4: "Recharge"}


def _num(v):
    """Human-friendly number: thousands separators, drop noise decimals."""
    v = float(v)
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:,.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def format_value(feature, value):
    """Business-friendly rendering of a raw feature value with units.

    Money -> "TK120"; volumes/minutes/days -> "120 MB"; one-hot & MyGP-active
    flags -> "Yes"/"No"; pack price (amount{n}) -> "TK39"; service encoding
    (srvc{n}) -> "code 4"; everything else -> a plain number.
    """
    f = feature.lower()

    # Binary flags: one-hot groups and MyGP monthly-active indicators.
    if f.startswith("circle_") or f.startswith("rchg_chnl_") or f in ("mygp_m1_act", "mygp_m2_act"):
        return "Yes" if float(value) >= 0.5 else "No"

    # Customer's top-N most-purchased packs.
    if f.startswith("amount") and f[6:].isdigit():
        return f"TK{_num(value)}"
    if f.startswith("srvc") and f[4:].isdigit():
        code = int(round(float(value)))
        return _SRVC_CODES.get(code, f"code {code}")

    # Strip lag suffix so lagged metrics inherit their base unit.
    root = f
    for suffix in ("_1", "_2"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break

    unit = _UNITS.get(root)
    if unit == "TK":
        return f"TK{_num(value)}"
    if unit:
        return f"{_num(value)} {unit}"
    return _num(value)


def load_scaler(model_kind):
    if model_kind not in _KIND:
        raise ValueError(f"model_kind must be one of {list(_KIND)}, got {model_kind!r}")
    with open(os.path.join(ARTIFACTS, _KIND[model_kind][1]), "rb") as f:
        return pickle.load(f)


def features_for(model_kind):
    """Authoritative feature list/order, read from the fitted scaler."""
    return list(load_scaler(model_kind).feature_names_in_)


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
def load_label_encoder():
    with open(os.path.join(ARTIFACTS, "label_encoder.pkl"), "rb") as f:
        return pickle.load(f)


def load_artifacts(model_kind):
    """Build the model, load its weights + scaler, and return everything the
    explainer/dashboard needs.

    Returns a dict: model, scaler, features, class_names, model_kind.
    """
    if model_kind not in _KIND:
        raise ValueError(f"model_kind must be one of {list(_KIND)}, got {model_kind!r}")
    weights_file, scaler_file = _KIND[model_kind]

    scaler = load_scaler(model_kind)
    features = list(scaler.feature_names_in_)

    model = build_model(input_shape=len(features))
    model.load_weights(os.path.join(ARTIFACTS, weights_file))

    class_names = list(load_label_encoder().classes_)
    return {
        "model": model,
        "scaler": scaler,
        "features": list(features),
        "class_names": class_names,
        "model_kind": model_kind,
    }


# --------------------------------------------------------------------------- #
# Class <-> denomination mapping
# class_names is sorted and aligned to the model's softmax output order.
# --------------------------------------------------------------------------- #
def deno_to_class_idx(class_names, deno):
    return class_names.index(int(deno))


def class_idx_to_deno(class_names, idx):
    return int(class_names[idx])


# --------------------------------------------------------------------------- #
# Explainer construction & SHAP computation
# --------------------------------------------------------------------------- #
def make_explainer(model, background_scaled, max_background=200):
    """GradientExplainer over a small background sample (scaled feature space).

    GradientExplainer is used rather than DeepExplainer because the network
    contains BatchNormalization layers, which DeepExplainer handles poorly under
    TF2.
    """
    import shap

    bg = np.asarray(background_scaled, dtype=np.float32)
    if len(bg) > max_background:
        rng = np.random.default_rng(42)
        bg = bg[rng.choice(len(bg), max_background, replace=False)]
    return shap.GradientExplainer(model, bg)


def _as_class_list(shap_values):
    """Normalize shap_values across SHAP versions to a list-of-(n, features),
    one entry per explained output."""
    if isinstance(shap_values, list):
        return shap_values
    arr = np.asarray(shap_values)
    # Newer SHAP returns (n, features, n_outputs) for multi-output models.
    if arr.ndim == 3:
        return [arr[:, :, j] for j in range(arr.shape[2])]
    return [arr]


def shap_top1(explainer, X_scaled):
    """SHAP values for each row's top-ranked output (the recommended pack).

    Efficient for the global precompute: only one class is explained per row.
    Returns (shap_matrix (n, features), class_indexes (n,)).
    """
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    shap_values, indexes = explainer.shap_values(X_scaled, ranked_outputs=1)
    sv = _as_class_list(shap_values)[0]
    idx = np.asarray(indexes).reshape(-1)
    return np.asarray(sv), idx


def shap_for_class(explainer, x_scaled_row, class_idx):
    """SHAP values for a single subscriber and a specific class (pack).

    x_scaled_row may be shape (features,) or (1, features). Returns a
    (features,) vector. Used by the on-demand per-subscriber waterfall, where
    the user may inspect any of the top-5 packs, not just the top-1.
    """
    x = np.asarray(x_scaled_row, dtype=np.float32).reshape(1, -1)
    shap_values = explainer.shap_values(x)
    per_class = _as_class_list(shap_values)
    return np.asarray(per_class[class_idx])[0]


def shap_for_class_batch(explainer, X_scaled, class_idx):
    """SHAP values for a batch of subscribers, all explained against ONE class.

    X_scaled is shape (n, features). Returns an (n, features) matrix. Unlike
    shap_top1 (which explains each row's own top-ranked pack), every row here is
    explained against the same `class_idx`, so the pack-scoped dependence view
    can show how a feature pushes one specific denomination across customers.
    """
    X = np.asarray(X_scaled, dtype=np.float32)
    shap_values = explainer.shap_values(X)
    per_class = _as_class_list(shap_values)
    return np.asarray(per_class[class_idx])


def expected_value_for_class(explainer, model, background_scaled, class_idx):
    """Base value (E[f(x)]) for a class = mean model output over the background.

    GradientExplainer does not always expose expected_value, so compute it from
    the background predictions to anchor the waterfall.
    """
    bg = np.asarray(background_scaled, dtype=np.float32)
    preds = model.predict(bg, verbose=0)
    return float(preds[:, class_idx].mean())


# --------------------------------------------------------------------------- #
# Single-subscriber preparation (on-demand local explanations)
# --------------------------------------------------------------------------- #
def prepare_single_subscriber(raw_rows, model_kind):
    """Run the existing preprocessing on one subscriber's raw row(s) and return
    a single-row DataFrame aligned to the model's feature list.

    Single-row pd.get_dummies inside preprocess_data won't emit every
    circle_*/rchg_chnl_* category, so we reindex to the full feature list and
    fill the missing one-hot columns with 0.
    """
    df = raw_rows.copy()
    df.columns = df.columns.str.lower()

    # The INFER tables don't store `arpu_total`, but preprocess_data uses it to
    # build revenue-share / rev-per-day features. Those engineered features are
    # NOT among the deployed models' inputs (they get dropped by the reindex
    # below), so an exact value is unnecessary — inject a proxy (sum of revenue
    # components) purely to keep preprocess_data from raising KeyError.
    if "arpu_total" not in df.columns:
        rev_cols = [c for c in ("voicerev_total", "datarev_total", "mixed_bundle_rev")
                    if c in df.columns]
        df["arpu_total"] = df[rev_cols].sum(axis=1) if rev_cols else 0.0

    processed = preprocess_data(df, infer=True)
    features = features_for(model_kind)
    aligned = processed.reindex(columns=features, fill_value=0)
    return aligned.astype(np.float32)
