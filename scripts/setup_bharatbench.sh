#!/usr/bin/env bash
# Clone the official BharatBench evaluators (pinned) and create their own Python 3.12 venv.
# The evaluators use the legacy openai==0.28 SDK, so they get a separate environment
# from the training/inference venv.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BB_COMMIT=${BB_COMMIT:-de5aa6beaaf25687d53cbbb284fcccea29a626a8}
PY=${PYTHON:-python3.12}

mkdir -p "$ROOT/third_party"
if [[ ! -d "$ROOT/third_party/BharatBench" ]]; then
  git clone https://github.com/ola-krutrim/BharatBench.git "$ROOT/third_party/BharatBench"
fi
git -C "$ROOT/third_party/BharatBench" checkout -q "$BB_COMMIT"

if [[ ! -d "$ROOT/.venv-bharatbench" ]]; then
  "$PY" -m venv "$ROOT/.venv-bharatbench"
fi
"$ROOT/.venv-bharatbench/bin/pip" install -q --upgrade pip
"$ROOT/.venv-bharatbench/bin/pip" install -q -r "$ROOT/requirements-bharatbench.txt"
echo "BharatBench ready at third_party/BharatBench (commit $BB_COMMIT)"
