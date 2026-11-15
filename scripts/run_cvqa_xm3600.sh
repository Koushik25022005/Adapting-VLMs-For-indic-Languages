#!/usr/bin/env bash
# Evaluate the zero-shot base model and the QLoRA adapter on CVQA and XM3600, then write the
# base-vs-QLoRA tables into the report. Neither benchmark needs a judge model or an API key.
#
#   bash scripts/run_cvqa_xm3600.sh
#   ADAPTER=path/to/final_adapter BENCHMARKS=cvqa bash scripts/run_cvqa_xm3600.sh
#
# Env vars:
#   ADAPTER     LoRA adapter dir (default: outputs/qwen25vl7b-chitrakshara-qlora/final_adapter)
#   BENCHMARKS  any of "cvqa xm3600" (default: both)
#   SKIP_BASE   1 = reuse the base model's existing score files instead of re-running it
#   LIMIT       debug: only the first N questions/images per language
#   EXTRA_ARGS  extra flags for both evaluators, e.g. "--batch-size 4"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
ADAPTER=${ADAPTER:-outputs/qwen25vl7b-chitrakshara-qlora/final_adapter}
BENCHMARKS=${BENCHMARKS:-"cvqa xm3600"}
SKIP_BASE=${SKIP_BASE:-0}
BASE_RUN=qwen25vl7b-base
OURS_RUN=qwen25vl7b-qlora
PY=.venv/bin/python

[[ -d .venv ]] || bash scripts/setup_env.sh
[[ -d "$ADAPTER" ]] || { echo "adapter not found: $ADAPTER (set ADAPTER=...)" >&2; exit 1; }
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
[[ -z "${LIMIT:-}" ]] || EXTRA+=(--limit "$LIMIT")

REPORT_ARGS=()
for BENCH in $BENCHMARKS; do
  case "$BENCH" in cvqa|xm3600) ;; *) echo "unknown benchmark: $BENCH" >&2; exit 1 ;; esac
  if [[ "$SKIP_BASE" != 1 ]]; then
    $PY -m "indicvlm.eval_$BENCH" --load-4bit --run-name "$BASE_RUN" "${EXTRA[@]}"
  fi
  $PY -m "indicvlm.eval_$BENCH" --load-4bit --run-name "$OURS_RUN" --adapter "$ADAPTER" "${EXTRA[@]}"
  REPORT_ARGS+=("--$BENCH-ours" "results/$OURS_RUN/${BENCH}_scores.json")
  if [[ -f "results/$BASE_RUN/${BENCH}_scores.json" ]]; then
    REPORT_ARGS+=("--$BENCH-base" "results/$BASE_RUN/${BENCH}_scores.json")
  fi
done
$PY -m indicvlm.update_report "${REPORT_ARGS[@]}"
