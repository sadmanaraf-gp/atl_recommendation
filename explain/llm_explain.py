"""LLM narrative layer for the pack-recommendation explainability dashboard.

Turns the structured SHAP evidence (the same feature / value / typical / effect
triples that ``dashboard.app.narrative`` renders as bullets) into a fluent,
business-readable paragraph. The model is instructed to reason ONLY from the
supplied facts and never to invent numbers, so every claim stays grounded in the
SHAP output.

Served by a local vLLM OpenAI-compatible server (see ``scripts/serve_llm.sh``).
Configuration comes from the project ``.env`` (mirrors the ORACLE_* pattern):

    LLM_BASE_URL   e.g. http://127.0.0.1:8000/v1   (unset -> LLM layer disabled)
    LLM_MODEL      e.g. qwen2.5-7b
    LLM_API_KEY    e.g. EMPTY   (vLLM ignores the value but the client needs one)

Every entry point degrades gracefully: on any missing config, connection error,
or timeout, ``explain`` returns ``None`` so callers fall back to the mechanical
bullet narrative and no view ever breaks.
"""
import os

import numpy as np
from dotenv import load_dotenv

from explain import shap_utils as su

# Load LLM_* (and ORACLE_*) from the project .env so both the batch job and the
# dashboard see the same config even when the caller has not called load_dotenv.
load_dotenv(os.path.join(su.ROOT, ".env"))

# Reuse the dashboard's segment labels without importing streamlit.
_KIND_LABELS = {
    "taker": "existing pack buyer (Taker)",
    "non_taker": "low-engagement customer (Non-Taker)",
}

_SYSTEM_PROMPT = (
    "You are a telecom analytics assistant explaining, to a business user, why a "
    "recommendation model suggested a specific mobile pack (a 'deno', priced in "
    "Bangladeshi Taka) to a customer.\n"
    "You are given the strongest drivers behind the recommendation. Each driver "
    "lists the customer's actual value, the typical value among buyers of this "
    "pack, and whether it RAISED or LOWERED this pack's score.\n"
    "Rules:\n"
    "- Explain the recommendation in 3-4 plain sentences a non-technical manager "
    "understands.\n"
    "- Use ONLY the numbers and facts provided. Never invent values, features, or "
    "reasons that are not in the drivers list.\n"
    "- Each driver ends with 'raised' or 'lowered'. This is the effect on the "
    "score and is GIVEN — copy it exactly. Do NOT infer the effect from whether "
    "the value is high or low; a high value can still lower the score. If a driver "
    "says 'lowered', you must say it lowered the score, never raised.\n"
    "- Do not merge drivers with different effects into one clause; keep each "
    "driver's stated effect attached to that driver.\n"
    "- Do not output bullet points, headings, or a preamble; write one short "
    "paragraph."
)


def is_configured():
    """True when an LLM endpoint is configured (LLM_BASE_URL is set)."""
    return bool(os.getenv("LLM_BASE_URL"))


def _client():
    from openai import OpenAI  # imported lazily so the dep is optional

    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        max_retries=1,
    )


def _drivers(feats, shap_vec, value_vec, ref_values, top_n):
    """Top-N |SHAP| drivers as human-readable dicts, reusing the dashboard's
    friendly_name / format_value helpers so the model sees business labels."""
    shap_vec = np.asarray(shap_vec, dtype=float)
    order = np.argsort(np.abs(shap_vec))[::-1][:top_n]
    out = []
    for i in order:
        feat = feats[i]
        out.append({
            "factor": su.friendly_name(feat),
            "customer_value": su.format_value(feat, value_vec[i]),
            "typical_for_pack": su.format_value(feat, ref_values[i]),
            "effect": "raised" if shap_vec[i] > 0 else "lowered",
        })
    return out


def build_messages(kind, deno, confidence, feats, shap_vec, value_vec,
                   ref_values, top_n=6):
    """Chat messages for the recommendation of `deno` to one customer."""
    drivers = _drivers(feats, shap_vec, value_vec, ref_values, top_n)
    lines = [
        f"Customer segment: {_KIND_LABELS.get(kind, kind)}",
        f"Recommended pack: TK{int(deno)}",
    ]
    if confidence is not None:
        lines.append(f"Model confidence: {float(confidence):.0%}")
    lines.append("\nDrivers (strongest first):")
    for d in drivers:
        lines.append(
            f"- {d['factor']}: customer = {d['customer_value']}, "
            f"typical for this pack = {d['typical_for_pack']} "
            f"-> {d['effect']} this pack's score"
        )
    lines.append(
        "\nWrite the explanation now, using only the facts above."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def explain(kind, deno, feats, shap_vec, value_vec, ref_values,
            confidence=None, top_n=6, max_tokens=220, temperature=0.2):
    """Generate a natural-language explanation, or None on any failure.

    Never raises: callers can wrap the result with ``or narrative(...)`` to fall
    back to the mechanical bullets.
    """
    if not is_configured():
        return None
    try:
        messages = build_messages(
            kind, deno, confidence, feats, shap_vec, value_vec, ref_values, top_n
        )
        resp = _client().chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen2.5-7b"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None
