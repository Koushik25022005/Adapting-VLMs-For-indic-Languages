#!/usr/bin/env bash
# End-to-end: data -> QLoRA training -> evaluation (base + fine-tuned) -> report.
#
#   export HF_TOKEN=...        # if Chitrakshara-Cap / Qwen are gated for your account
#   bash scripts/run_pipeline.sh
#   STREAMING=1 bash scripts/run_pipeline.sh   # stream Chitrakshara during training, no local copy
#
# BENCHMARKS picks the evaluations (default "cvqa xm3600"; no API key needed). Add
# "bharatbench" for the GPT-judged BharatBench comparison with Chitrarth; that one needs
# OPENAI_API_KEY.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PER_LANG=${PER_LANG:-5000}
STREAMING=${STREAMING:-0}
if [[ "$STREAMING" == 1 ]]; then
  CONFIG=${CONFIG:-configs/qlora_qwen25vl_chitrakshara_stream.yaml}
else
  CONFIG=${CONFIG:-configs/qlora_qwen25vl_chitrakshara.yaml}
fi
ADAPTER=${ADAPTER:-outputs/qwen25vl7b-chitrakshara-qlora/final_adapter}
SKIP_BASE=${SKIP_BASE:-0}
BENCHMARKS=${BENCHMARKS:-"cvqa xm3600"}

[[ -d .venv ]] || bash scripts/setup_env.sh
PY=.venv/bin/python

# 1) Data
[[ "$STREAMING" == 1 || -f data/chitrakshara_cap/train.jsonl ]] || \
  $PY -m indicvlm.prepare_data --out data/chitrakshara_cap --per-language "$PER_LANG"

# 2) QLoRA training
[[ -d "$ADAPTER" ]] || $PY -m indicvlm.train --config "$CONFIG"

# 3) CVQA / XM3600: zero-shot base model vs QLoRA
NEW=$(tr ' ' '\n' <<< "$BENCHMARKS" | grep -E '^(cvqa|xm3600)$' | xargs || true)
if [[ -n "$NEW" ]]; then
  ADAPTER="$ADAPTER" BENCHMARKS="$NEW" SKIP_BASE="$SKIP_BASE" bash scripts/run_cvqa_xm3600.sh
fi

# 4) BharatBench (optional): responses, official scoring, comparison with Chitrarth
if [[ " $BENCHMARKS " == *" bharatbench "* ]]; then
  [[ -d third_party/BharatBench ]] || bash scripts/setup_bharatbench.sh
  [[ -d data/bharatbench_images ]] || $PY scripts/download_bharatbench_images.py --out data/bharatbench_images
  $PY -m indicvlm.infer_bharatbench --adapter "$ADAPTER" --load-4bit \
    --out results/responses/qwen25vl7b-qlora --model-name qwen25vl7b-qlora
  bash scripts/run_bharatbench.sh results/responses/qwen25vl7b-qlora qwen25vl7b-qlora
  BASE_ARG=()
  if [[ "$SKIP_BASE" != 1 ]]; then
    $PY -m indicvlm.infer_bharatbench --load-4bit \
      --out results/responses/qwen25vl7b-base --model-name qwen25vl7b-base
    bash scripts/run_bharatbench.sh results/responses/qwen25vl7b-base qwen25vl7b-base
    BASE_ARG=(--base results/qwen25vl7b-base/bharatbench_scores.json)
  fi
  $PY -m indicvlm.update_report --ours results/qwen25vl7b-qlora/bharatbench_scores.json "${BASE_ARG[@]}"
fi
