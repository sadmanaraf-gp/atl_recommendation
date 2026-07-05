"""Explainable-AI dashboard for the ATL pack-recommendation models.

    streamlit run dashboard/app.py

Four views:
  1. Global Drivers      - what drives recommendations overall (precomputed SHAP)
  2. Subscriber Lookup   - why THIS subscriber got THESE packs (on-demand SHAP)
  3. Feature Dependence  - how one feature moves pack probability
  4. Model Health        - hit@k / NDCG / revenue capture + warnings (eval_metrics.json)

Each view leads with a plain-language summary; raw SHAP detail sits behind an
expander so the dashboard serves both business and data-science audiences.
"""
import os
import sys
import json
import gzip
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Make `scripts` and `explain` importable regardless of launch directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load ORACLE_* credentials from the project .env so live subscriber lookup works
# (mirrors predict.py / train.py). Without this, os.getenv returns None and the
# "Live Oracle" mode stays hidden.
load_dotenv(os.path.join(ROOT, ".env"))

from explain import shap_utils as su  # noqa: E402

KIND_LABELS = {"taker": "Taker (existing pack buyers)",
               "non_taker": "Non-Taker (low-engagement)"}

st.set_page_config(page_title="ATL Reco Explainability", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_global(model_kind):
    path = os.path.join(su.SHAP_CACHE, f"global_{model_kind}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_sample_features(model_kind):
    path = os.path.join(su.SHAP_CACHE, f"sample_features_{model_kind}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_eval_metrics():
    path = os.path.join(su.ARTIFACTS, "eval_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=True)
def get_model_bundle(model_kind):
    """Model + scaler + explainer (background from the cached sample)."""
    art = su.load_artifacts(model_kind)
    sample = load_sample_features(model_kind)
    if sample is None:
        raise FileNotFoundError(
            f"Missing sample cache for {model_kind}. Run: python -m explain.precompute_shap"
        )
    bg = art["scaler"].transform(
        sample[art["features"]].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    art["explainer"] = su.make_explainer(art["model"], bg)
    art["background_scaled"] = bg[: min(200, len(bg))]
    return art


def shap_columns(df, features):
    return df[[f"shap_{c}" for c in features]].to_numpy()


def val_columns(df, features):
    return df[[f"val_{c}" for c in features]].to_numpy()


@st.cache_data(show_spinner=False)
def pack_reference(model_kind, deno, min_rows=20):
    """Typical (median) attribute values among sampled customers whose top-1
    recommendation is this pack. Falls back to the population median when the
    pack has too few sampled buyers. Returns a dict {feature: median_value}.
    Uses the existing global parquet cache — no model calls.
    """
    g = load_global(model_kind)
    features = su.features_for(model_kind)
    if g is None:
        return {}
    subset = g[g["deno"] == int(deno)]
    if len(subset) < min_rows:
        subset = g  # not enough pack-specific samples -> population median
    med = subset[[f"val_{c}" for c in features]].median()
    return {c: float(med[f"val_{c}"]) for c in features}


# --------------------------------------------------------------------------- #
# Plain-language narrative
# --------------------------------------------------------------------------- #
def narrative(features, shap_vec, value_vec, ref_values, top_n=5):
    """Plain-language driver phrases from the strongest SHAP contributions.

    Each bullet keeps two distinct facts separate to avoid confusion:
      - the *value*: this customer's actual attribute value, with the typical
        value for buyers of this pack shown for comparison;
      - the *effect*: whether it raised or lowered THIS pack's score, which is
        the sign of the SHAP value — not the size of the attribute itself.
    """
    order = np.argsort(np.abs(shap_vec))[::-1][:top_n]
    bullets = []
    for i in order:
        feat = features[i]
        name = su.friendly_name(feat)
        effect = "raised" if shap_vec[i] > 0 else "lowered"
        cust = su.format_value(feat, value_vec[i])
        typ = su.format_value(feat, ref_values[i])
        bullets.append(
            f"**{name}** = {cust} (typical ≈ {typ}) → {effect} this pack's score"
        )
    return bullets


def contribution_chart(features, shap_vec, value_vec=None, top_n=12):
    """Signed horizontal bar of the strongest contributions (waterfall-style).

    When value_vec is given, each bar is labelled with the attribute's value
    (e.g. "Data revenue = ৳120") so the chart is readable on its own.
    """
    order = np.argsort(np.abs(shap_vec))[::-1][:top_n]

    def _label(i):
        name = su.friendly_name(features[i])
        if value_vec is not None:
            return f"{name} = {su.format_value(features[i], value_vec[i])}"
        return name

    names = [_label(i) for i in order][::-1]
    vals = [float(shap_vec[i]) for i in order][::-1]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(names) + 1))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to this pack's probability")
    ax.set_title("Top drivers for this recommendation")
    fig.tight_layout()
    return fig


def help_box(body_md):
    """Collapsed, self-serve 'how to read this' explainer for a view."""
    with st.expander("❓ How to read this"):
        st.markdown(body_md)


# --------------------------------------------------------------------------- #
# View 1: Global Drivers
# --------------------------------------------------------------------------- #
def view_global():
    st.header("Global Drivers")
    st.caption("Which subscriber attributes drive pack recommendations across the population.")
    help_box(
        "- **Bar chart (feature importance):** the attributes the model leans on most, "
        "overall. Longer bar = bigger influence on which pack gets recommended.\n"
        "- **Beeswarm (technical detail):** each dot is one sampled customer. Dots to the "
        "**right** pushed the recommended pack **up**; to the **left**, down. Dot **color** "
        "is the attribute's value for that customer (red = high, blue = low) — so you can see, "
        "e.g., whether *high* recharge amount tends to push a pack up.\n"
        "- Use the **pack filter** to ask *what drives this specific denomination?*"
    )

    kind = st.radio("Model", list(KIND_LABELS), format_func=KIND_LABELS.get,
                    horizontal=True, key="global_kind")
    g = load_global(kind)
    if g is None:
        st.warning("No SHAP cache found. Run `python -m explain.precompute_shap` first.")
        return
    features = su.features_for(kind)

    denos = sorted(g["deno"].unique())
    pick = st.selectbox("Focus on a specific recommended pack (denomination)?",
                        ["All packs"] + [str(d) for d in denos])
    gg = g if pick == "All packs" else g[g["deno"] == int(pick)]
    if gg.empty:
        st.info("No sampled subscribers for that pack.")
        return

    shap_mat = shap_columns(gg, features)
    mean_abs = np.abs(shap_mat).mean(axis=0)
    top = np.argsort(mean_abs)[::-1][:5]

    st.subheader("In plain language")
    st.markdown(
        "Top 5 reasons this model recommends "
        + (f"pack {pick}" if pick != "All packs" else "packs")
        + ":\n"
        + "\n".join(f"- **{su.friendly_name(features[i])}**" for i in top)
    )

    imp = (pd.DataFrame({"feature": [su.friendly_name(features[i]) for i in range(len(features))],
                         "importance": mean_abs})
           .sort_values("importance", ascending=False).head(20))
    st.plotly_chart(
        px.bar(imp.sort_values("importance"), x="importance", y="feature",
               orientation="h", title="Mean |SHAP| — overall feature importance",
               height=600),
        use_container_width=True,
    )

    with st.expander("Technical detail: SHAP beeswarm"):
        import shap
        fig = plt.figure()
        shap.summary_plot(shap_mat, val_columns(gg, features),
                          feature_names=[su.friendly_name(f) for f in features],
                          show=False, max_display=20)
        st.pyplot(fig, clear_figure=True)
        st.caption(f"Based on {len(gg):,} sampled subscribers; SHAP for each subscriber's "
                   "top-ranked (recommended) pack.")


# --------------------------------------------------------------------------- #
# View 2: Subscriber Lookup
# --------------------------------------------------------------------------- #
def whatif_panel(bundle, kind, feats, x_unscaled, base_top5, choice, class_idx,
                 base_shap, ref):
    """Interactive 'what-if': adjust the top numeric drivers and see how this
    customer's recommendations shift. Re-runs the model live (cheap for one
    row); SHAP recompute is gated behind a button (~1-2s)."""
    st.subheader("🔧 What-if: test a change")
    st.caption("Drag a slider to change an attribute and see how the recommendation "
               "shifts. Region and service-code attributes are held fixed.")

    sample = load_sample_features(kind)

    # Expose the strongest numeric drivers for this pack (skip one-hot/service-code).
    exposed = []
    for i in np.argsort(np.abs(base_shap))[::-1]:
        f = feats[i].lower()
        if f.startswith(("circle_", "rchg_chnl_")) or (f.startswith("srvc") and f[4:].isdigit()):
            continue
        exposed.append(int(i))
        if len(exposed) >= 8:
            break

    # Reset must clear widget keys *before* the sliders are instantiated below.
    if st.button("Reset to actual values", key="wi_reset"):
        for i in exposed:
            st.session_state.pop(f"wi_{feats[i]}", None)

    modified = x_unscaled.astype(np.float32).copy()
    cols = st.columns(2)
    for n, i in enumerate(exposed):
        feat = feats[i]
        cur = float(x_unscaled[i])
        hi = float(sample[feat].quantile(0.99)) if sample is not None else cur * 2
        hi = max(hi, cur, 1.0)
        with cols[n % 2]:
            modified[i] = st.slider(
                f"{su.friendly_name(feat)}  (now {su.format_value(feat, cur)})",
                min_value=0.0, max_value=float(round(hi, 2)),
                value=float(min(max(cur, 0.0), hi)),
                step=float(max(hi / 100.0, 0.01)), key=f"wi_{feat}",
            )

    x_mod = bundle["scaler"].transform(modified.reshape(1, -1)).astype(np.float32)[0]
    new_top5 = predict_top5(bundle, x_mod)
    base_p = float(base_top5.loc[base_top5["_class_idx"] == class_idx, "probability"].iloc[0])
    new_p = float(bundle["model"].predict(x_mod.reshape(1, -1), verbose=0)[0][class_idx])

    c1, c2 = st.columns([1, 2])
    c1.metric(f"Pack {choice} probability", f"{new_p:.3f}", delta=f"{new_p - base_p:+.3f}")
    c2.caption("New top-5 after your changes")
    c2.dataframe(new_top5.drop(columns="_class_idx").style.format({"probability": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    if st.button("Recompute explanation for changed values", key="wi_explain"):
        with st.spinner("Recomputing SHAP for the changed values..."):
            new_shap = su.shap_for_class(bundle["explainer"], x_mod, class_idx)
        for b in narrative(feats, new_shap, modified, ref):
            st.markdown(f"- {b}")
        st.pyplot(contribution_chart(feats, new_shap, modified), clear_figure=True)


def fetch_subscriber_from_oracle(msisdn):
    """Return (raw_row_df, pack_flag) for one MSISDN, or (None, None).

    The three INFER tables are ~50M rows with no index on MSISDN, so a plain
    single-row lookup full-scans each (~180s total). We cut that to ~25s by
    (a) adding a `/*+ PARALLEL(8) */` hint (the prep pipeline already relies on
    parallel scans) and (b) running the three scans concurrently on separate
    connections. `read_oracle_data` opens/closes its own connection per call,
    so it is safe to call from worker threads.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scripts.db_utils import read_oracle_data
    from scripts import config as cfg

    db = {"user": os.getenv("ORACLE_USER"), "password": os.getenv("ORACLE_PASSWORD"),
          "dsn": os.getenv("ORACLE_DSN")}
    if not all(db.values()):
        return None, None

    m = int(msisdn)  # cast guards against SQL injection in the f-string
    specs = [
        (cfg.TABLE_INFER_01, "MSISDN"),
        (cfg.TABLE_INFER_02, "MSISDN_1"),
        (cfg.TABLE_INFER_03, "MSISDN_2"),
    ]

    def _query(spec):
        table, col = spec
        sql = f"SELECT /*+ PARALLEL(8) */ * FROM {table} WHERE {col} = {m}"
        return read_oracle_data(sql, **db)

    with ThreadPoolExecutor(max_workers=3) as ex:
        b1, b2, b3 = list(ex.map(_query, specs))

    if b1 is None or b1.empty:
        return None, None
    b1["MSISDN"] = b1["MSISDN"].astype(int)
    raw = b1
    if b2 is not None and not b2.empty:
        raw = raw.merge(b2, "left", left_on="MSISDN", right_on="MSISDN_1")
    if b3 is not None and not b3.empty:
        raw = raw.merge(b3, "left", left_on="MSISDN", right_on="MSISDN_2")

    flag = None
    for c in raw.columns:
        if c.lower() == "pack_flag":
            flag = str(raw.iloc[0][c])
            break
    return raw, flag


def predict_top5(bundle, x_scaled_row):
    proba = bundle["model"].predict(x_scaled_row.reshape(1, -1), verbose=0)[0]
    class_names = bundle["class_names"]
    k = min(5, len(class_names))
    top_idx = np.argsort(proba)[::-1][:k]
    return pd.DataFrame({
        "rank": np.arange(1, k + 1),
        "pack (deno)": [su.class_idx_to_deno(class_names, i) for i in top_idx],
        "probability": [float(proba[i]) for i in top_idx],
        "_class_idx": top_idx,
    })


def view_subscriber():
    st.header("Subscriber Lookup")
    st.caption("Why a specific subscriber received their top-5 pack recommendations.")
    help_box(
        "- **Top-5 table:** the model's ranked pack picks for this customer. "
        "**Probability** is the model's confidence — higher means a stronger recommendation.\n"
        "- **In plain language:** each driver shows the customer's **actual value** next to "
        "what's **typical for buyers of that pack**, then whether it **raised** or **lowered** "
        "this pack's score. Value and effect are separate: a value can be high yet still lower "
        "the score.\n"
        "- **Contribution chart:** **green** bars raised this pack's chance, **red** lowered it; "
        "each bar is labelled with the attribute's value. Longest bar = strongest reason.\n"
        "- **What-if:** drag a slider to change an attribute and watch the recommendation shift."
    )

    oracle_ready = all(os.getenv(k) for k in ("ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_DSN"))
    mode = st.radio(
        "Source",
        ["Sample (offline)", "Live Oracle"] if oracle_ready else ["Sample (offline)"],
        horizontal=True,
    )

    if mode.startswith("Sample"):
        kind = st.radio("Model", list(KIND_LABELS), format_func=KIND_LABELS.get,
                        horizontal=True, key="sub_kind")
        sample = load_sample_features(kind)
        if sample is None:
            st.warning("No sample cache. Run `python -m explain.precompute_shap` first.")
            return
        msisdn = st.selectbox("MSISDN", sample["msisdn"].astype(str).tolist()[:2000])
        row = sample[sample["msisdn"].astype(str) == str(msisdn)].iloc[[0]]
        feats = su.features_for(kind)
        x_unscaled = row[feats].to_numpy(dtype=np.float32)[0]
    else:
        msisdn = st.text_input("Enter MSISDN")
        if not msisdn:
            st.info("Enter an MSISDN to fetch and explain.")
            return
        with st.spinner("Fetching subscriber from Oracle..."):
            raw, flag = fetch_subscriber_from_oracle(msisdn)
        if raw is None:
            st.error("Subscriber not found (or Oracle unavailable).")
            return
        kind = {"TAKER": "taker", "NON_TAKER": "non_taker"}.get(str(flag).upper(), "taker")
        st.caption(f"pack_flag = {flag} → {KIND_LABELS[kind]} model")
        aligned = su.prepare_single_subscriber(raw, kind)
        feats = su.features_for(kind)
        x_unscaled = aligned[feats].to_numpy(dtype=np.float32)[0]

    bundle = get_model_bundle(kind)
    x_scaled = bundle["scaler"].transform(x_unscaled.reshape(1, -1)).astype(np.float32)[0]

    top5 = predict_top5(bundle, x_scaled)
    st.subheader("Top-5 recommended packs")
    st.dataframe(top5.drop(columns="_class_idx").style.format({"probability": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    choice = st.selectbox("Explain which pack?",
                          top5["pack (deno)"].tolist())
    class_idx = int(top5[top5["pack (deno)"] == choice]["_class_idx"].iloc[0])

    with st.spinner("Computing SHAP for this subscriber..."):
        shap_vec = su.shap_for_class(bundle["explainer"], x_scaled, class_idx)

    # Typical attribute values among customers whose top pick is this pack.
    ref_map = pack_reference(kind, choice)
    ref = np.array([ref_map.get(c, 0.0) for c in feats], dtype=np.float32)

    st.subheader("In plain language")
    st.markdown(
        f"Biggest factors behind Pack **{choice}**'s score for this customer "
        "(actual value vs. what's typical for this pack's buyers → **raised** or "
        "**lowered** its ranking):"
    )
    for b in narrative(feats, shap_vec, x_unscaled, ref):
        st.markdown(f"- {b}")

    st.pyplot(contribution_chart(feats, shap_vec, x_unscaled), clear_figure=True)

    with st.expander("Technical detail: feature values & raw SHAP"):
        detail = pd.DataFrame({
            "feature": [su.friendly_name(c) for c in feats],
            "value": [su.format_value(c, v) for c, v in zip(feats, x_unscaled)],
            "typical (this pack)": [su.format_value(c, r) for c, r in zip(feats, ref)],
            "shap": shap_vec,
        }).reindex(np.argsort(np.abs(shap_vec))[::-1])
        st.dataframe(detail, use_container_width=True, hide_index=True)

    whatif_panel(bundle, kind, feats, x_unscaled, top5, choice, class_idx, shap_vec, ref)


# --------------------------------------------------------------------------- #
# View 3: Feature Dependence
# --------------------------------------------------------------------------- #
def view_dependence():
    st.header("Feature Dependence")
    st.caption("How the value of one feature moves its contribution to the recommended pack.")
    help_box(
        "- Each dot is one sampled customer.\n"
        "- **X-axis:** the chosen attribute's actual value. **Y-axis:** how much that attribute "
        "pushed the recommended pack up or down (its SHAP effect).\n"
        "- Dots **above** the dashed line **increased** the pack's probability for that customer; "
        "**below** the line, **decreased** it.\n"
        "- The overall shape shows the relationship — e.g. rising left-to-right means *more of "
        "this attribute → stronger push toward the pack*.\n"
        "- **Color by** a second attribute to spot interactions between two factors."
    )

    kind = st.radio("Model", list(KIND_LABELS), format_func=KIND_LABELS.get,
                    horizontal=True, key="dep_kind")
    g = load_global(kind)
    if g is None:
        st.warning("No SHAP cache found. Run `python -m explain.precompute_shap` first.")
        return
    features = su.features_for(kind)

    feat = st.selectbox("Feature", features,
                        format_func=lambda f: f"{su.friendly_name(f)}  ({f})")
    color_feat = st.selectbox("Color by (interaction)", ["(none)"] + features,
                              format_func=lambda f: su.friendly_name(f) if f != "(none)" else f)

    plot_df = pd.DataFrame({
        "value": g[f"val_{feat}"],
        "shap": g[f"shap_{feat}"],
    })
    kwargs = {}
    if color_feat != "(none)":
        plot_df["color"] = g[f"val_{color_feat}"]
        kwargs = {"color": "color", "color_continuous_scale": "RdBu"}
    fig = px.scatter(plot_df, x="value", y="shap", opacity=0.6,
                     labels={"value": su.friendly_name(feat),
                             "shap": f"SHAP value for {su.friendly_name(feat)}",
                             "color": su.friendly_name(color_feat) if color_feat != "(none)" else ""},
                     title=f"Dependence: {su.friendly_name(feat)}", **kwargs)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points above the dashed line: this feature increased the recommended pack's "
               "probability for that subscriber; below: decreased it.")


# --------------------------------------------------------------------------- #
# View 4: Model Health
# --------------------------------------------------------------------------- #
def view_health():
    st.header("Model Health")
    help_box(
        "- **Hit@1 / Hit@5:** how often the pack the customer actually bought was the model's "
        "top pick / somewhere in its top-5. Higher is better.\n"
        "- **NDCG@5:** ranking quality — rewards putting the right pack nearer the top of the 5.\n"
        "- **Revenue capture@5:** of the revenue the ideal recommendation could have captured, "
        "the share this model's top-5 captures. ~100% is on target.\n"
        "- **Warnings:** flag packs that are over-recommended (class collapse) or never "
        "recommended (dead packs) — worth investigating before trusting those packs."
    )
    metrics = load_eval_metrics()
    if not metrics:
        st.warning("artifacts/eval_metrics.json not found.")
        return
    for m in metrics:
        st.subheader(m.get("model_name", "model"))
        c = st.columns(4)
        c[0].metric("Hit@1", f"{m['hit_at_1']:.1%}")
        c[1].metric("Hit@5", f"{m['hit_at_5']:.1%}")
        c[2].metric("NDCG@5", f"{m['ndcg_at_5']:.3f}")
        c[3].metric("Revenue capture@5",
                    f"{m['revenue_at_5']['revenue_capture_rate']:.1%}")
        warns = m.get("class_distribution", {}).get("warnings", [])
        if warns:
            for w in warns:
                st.warning(w)
        else:
            st.success("No class-distribution warnings.")
        st.divider()


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def main():
    st.sidebar.title("ATL Reco Explainability")
    st.sidebar.caption("SHAP-based transparency for the pack-recommendation models.")
    st.sidebar.info(
        "**What is SHAP?** It scores how much each customer attribute pushed a pack "
        "**up or down** in the model's ranking. **Green = pushed up, red = pushed down.** "
        "Every view has a *❓ How to read this* box."
    )
    page = st.sidebar.radio(
        "View",
        ["Global Drivers", "Subscriber Lookup", "Feature Dependence", "Model Health"],
    )
    {"Global Drivers": view_global,
     "Subscriber Lookup": view_subscriber,
     "Feature Dependence": view_dependence,
     "Model Health": view_health}[page]()


# Streamlit runs this script as "__main__" on every interaction; the guard lets
# the helper functions be imported (e.g. for tests) without launching the UI.
if __name__ == "__main__":
    main()
