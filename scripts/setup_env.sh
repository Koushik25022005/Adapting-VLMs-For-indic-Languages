#!/usr/bin/env bash
# Create the Python 3.12 training/inference environment in .venv
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3.12}
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || { echo "Python 3.12 is required (set PYTHON=/path/to/python3.12)"; exit 1; }
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
