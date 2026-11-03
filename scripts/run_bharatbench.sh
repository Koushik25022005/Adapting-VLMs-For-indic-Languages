#!/usr/bin/env bash
# Score generated responses with the official BharatBench evaluators, then collect the
# numbers into results/<run-name>/bharatbench_scores.json.
#
#   OPENAI_API_KEY=... bash scripts/run_bharatbench.sh results/responses/qwen25vl7b-qlora qwen25vl7b-qlora
#
# Env vars:
#   LANGS        space-separated BharatBench language names (default: the 11 Chitrarth reports)
#   TASKS        any of "pope llavabench mmvet" (default: all)
#   POPE_SCORER  gpt (default, as in Chitrarth's README) | muril | labse | xlm-roberta | multilingual-bert
#   JUDGE_MODEL  override the GPT judge for POPE/LLaVA-Bench (default in BharatBench: gpt-4-0613).
#                Changing it makes the numbers not strictly comparable with Chitrarth's.
#   MMVET_JUDGE  MM-Vet judge (BharatBench default: gpt-4o-mini-2024-07-18)
set -euo pipefail

RESP_DIR=$(realpath "$1")
RUN_NAME=${2:-$(basename "$RESP_DIR")}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BB_SRC="$ROOT/third_party/BharatBench"
WORK="$ROOT/results/$RUN_NAME/bharatbench"
LANGS=${LANGS:-"Telugu Hindi Bengali Malayalam Kannada Assamese Tamil Marathi Gujarati Odia English"}
TASKS=${TASKS:-"pope llavabench mmvet"}
POPE_SCORER=${POPE_SCORER:-gpt}
MMVET_JUDGE=${MMVET_JUDGE:-gpt-4o-mini-2024-07-18}

[[ -d "$BB_SRC" ]] || bash "$ROOT/scripts/setup_bharatbench.sh"
if [[ "$POPE_SCORER" == gpt || "$TASKS" == *llavabench* || "$TASKS" == *mmvet* ]]; then
  : "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for GPT-judged BharatBench tasks}"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv-bharatbench/bin/activate"

# Fresh working copy per run (the evaluators append to / resume from their output files).
rm -rf "$WORK"; mkdir -p "$(dirname "$WORK")"
cp -r "$BB_SRC" "$WORK"; rm -rf "$WORK/.git"
for d in Indic-POPE-Evaluator/model_responses Indic-POPE-Evaluator/scores \
         Indic-LLaVABench-Evaluator/model_responses Indic-LLaVABench-Evaluator/answers \
         Indic-LLaVABench-Evaluator/reviews Indic-LLaVABench-Evaluator/scores \
         Indic-MMVet-Evaluator/model_responses Indic-MMVet-Evaluator/results Indic-MMVet-Evaluator/grades; do
  rm -rf "${WORK:?}/$d"; mkdir -p "$WORK/$d"
done
cp -r "$RESP_DIR"/. "$WORK"/

if [[ -n "${JUDGE_MODEL:-}" ]]; then
  sed -i "s/gpt-4-0613/$JUDGE_MODEL/" "$WORK/Indic-POPE-Evaluator/scripts/Indic_eval_pope.py" \
      "$WORK/Indic-LLaVABench-Evaluator/scripts/eval_gpt_review_bench.py"
fi

for LANG in $LANGS; do
  lower=$(echo "$LANG" | tr '[:upper:]' '[:lower:]')
  if [[ "$TASKS" == *pope* && -f "$WORK/Indic-POPE-Evaluator/model_responses/${lower}_results.jsonl" ]]; then
    echo "== POPE / $LANG ($POPE_SCORER)"
    pushd "$WORK/Indic-POPE-Evaluator" >/dev/null
    if [[ "$POPE_SCORER" == gpt ]]; then
      python scripts/Indic_eval_pope.py --language "$lower"
    else
      python scripts/Indic_eval_pope_embmodels.py --language "$lower" --model "$POPE_SCORER"
    fi
    popd >/dev/null
  fi
  if [[ "$TASKS" == *llavabench* && -f "$WORK/Indic-LLaVABench-Evaluator/model_responses/responses_${LANG}.jsonl" ]]; then
    echo "== LLaVA-Bench / $LANG"
    pushd "$WORK/Indic-LLaVABench-Evaluator" >/dev/null
    bash run_answer_files.sh "$LANG" && bash llavabench.sh "$LANG"
    popd >/dev/null
  fi
  if [[ "$TASKS" == *mmvet* && -f "$WORK/Indic-MMVet-Evaluator/model_responses/mm-vet_${LANG}.json" ]]; then
    echo "== MM-Vet / $LANG"
    pushd "$WORK/Indic-MMVet-Evaluator" >/dev/null
    python mm-vet_evaluator_all.py --lang "$LANG" --gpt_model "$MMVET_JUDGE"
    popd >/dev/null
  fi
done

deactivate
"$ROOT/.venv/bin/python" -m indicvlm.collect_scores --workdir "$WORK" --pope-scorer "$POPE_SCORER" \
  --model-name "$RUN_NAME" --out "$ROOT/results/$RUN_NAME/bharatbench_scores.json"
