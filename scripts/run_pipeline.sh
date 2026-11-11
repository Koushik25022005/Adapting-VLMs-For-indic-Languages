#!/usr/bin/env bash
# End-to-end: data -> QLoRA training -> BharatBench inference (base + fine-tuned) -> scoring -> report.
#
#   export HF_TOKEN=...        # if Chitrakshara-Cap / Qwen are gated for your account
#   export OPENAI_API_KEY=...  # BharatBench GPT judges
#   bash scripts/run_pipeline.sh
#   STREAMING=1 bash scripts/run_pipeline.sh   # stream Chitrakshara during training, no local copy
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

[[ -d .venv ]] || bash scripts/setup_env.sh
[[ -d third_party/BharatBench ]] || bash scripts/setup_bharatbench.sh
PY=.venv/bin/python

# 1) Data
[[ "$STREAMING" == 1 || -f data/chitrakshara_cap/train.jsonl ]] || \
  $PY -m indicvlm.prepare_data --out data/chitrakshara_cap --per-language "$PER_LANG"
[[ -d data/bharatbench_images ]] || $PY scripts/download_bharatbench_images.py --out data/bharatbench_images

# 2) QLoRA training
[[ -d "$ADAPTER" ]] || $PY -m indicvlm.train --config "$CONFIG"

# 3) BharatBench responses
$PY -m indicvlm.infer_bharatbench --adapter "$ADAPTER" --load-4bit \
  --out results/responses/qwen25vl7b-qlora --model-name qwen25vl7b-qlora
if [[ "$SKIP_BASE" != 1 ]]; then
  $PY -m indicvlm.infer_bharatbench --load-4bit \
    --out results/responses/qwen25vl7b-base --model-name qwen25vl7b-base
fi

# 4) Official BharatBench scoring
bash scripts/run_bharatbench.sh results/responses/qwen25vl7b-qlora qwen25vl7b-qlora
[[ "$SKIP_BASE" == 1 ]] || bash scripts/run_bharatbench.sh results/responses/qwen25vl7b-base qwen25vl7b-base

# 5) Report
BASE_ARG=(); [[ "$SKIP_BASE" == 1 ]] || BASE_ARG=(--base results/qwen25vl7b-base/bharatbench_scores.json)
$PY -m indicvlm.update_report --ours results/qwen25vl7b-qlora/bharatbench_scores.json "${BASE_ARG[@]}"
