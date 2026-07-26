#!/usr/bin/env bash
# Launch the local vLLM OpenAI-compatible server that generates the dashboard's
# pack-recommendation explanations (see explain/llm_explain.py).
#
# Runs Qwen2.5-7B-Instruct on the local L40S, sharing the GPU with the
# TensorFlow/Keras Streamlit app. --gpu-memory-utilization is kept well under 1.0
# so the SHAP GradientExplainer (TF) has headroom; the Streamlit app must enable
# TF memory growth (see dashboard/app.py) so TF does not grab the whole card.
#
#   bash scripts/serve_llm.sh
#
# Smoke test once it is up:
#   curl http://127.0.0.1:8000/v1/models
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/home/jovyan/hf_cache}"

exec "$ROOT/.venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.55}" \
  --max-model-len "${VLLM_MAX_LEN:-8192}" \
  --dtype float16
