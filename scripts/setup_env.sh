#!/usr/bin/env bash
# Create the Python 3.13.5 training/inference environment in .venv
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$(bash "$ROOT/scripts/find_python.sh")
"$PY" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q --upgrade pip
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/pip" install -q -e "$ROOT"
# Optional, speeds up training on Ampere+: pip install flash-attn --no-build-isolation
"$ROOT/.venv/bin/python" - <<'EOF'
import sys, torch, transformers, peft, bitsandbytes
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| bitsandbytes", bitsandbytes.__version__, "| CUDA", torch.cuda.is_available())
EOF
