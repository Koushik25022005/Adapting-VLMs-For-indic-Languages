#!/usr/bin/env bash
# Create the Python 3.13.5 training/inference environment in .venv
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$(bash "$ROOT/scripts/find_python.sh")
"$PY" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q --upgrade pip

# PyTorch build. The newest PyTorch on PyPI is built for CUDA 13 and reports "CUDA False" on
# NVIDIA drivers that only support CUDA 12.x. On such drivers install the CUDA 12.4 build.
#   TORCH_CUDA=cu124    force that build        TORCH_CUDA=default   take whatever PyPI gives
#   TORCH_VERSION / TORCHVISION_VERSION         versions used with TORCH_CUDA (default 2.6.0 / 0.21.0)
TORCH_CUDA=${TORCH_CUDA:-auto}
if [[ "$TORCH_CUDA" == auto ]]; then
  DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)
  case "$DRIVER_CUDA" in
    12.[4-9]*) TORCH_CUDA=cu124 ;;
    *) TORCH_CUDA=default ;;
  esac
  echo "NVIDIA driver supports CUDA ${DRIVER_CUDA:-unknown}; PyTorch build: $TORCH_CUDA"
fi
if [[ "$TORCH_CUDA" != default ]]; then
  "$ROOT/.venv/bin/pip" install -q "torch==${TORCH_VERSION:-2.6.0}" "torchvision==${TORCHVISION_VERSION:-0.21.0}" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
fi
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/pip" install -q -e "$ROOT"
# Optional, speeds up training on Ampere+: pip install flash-attn --no-build-isolation
"$ROOT/.venv/bin/python" - <<'EOF'
import sys, torch, transformers, peft, bitsandbytes
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| bitsandbytes", bitsandbytes.__version__, "| CUDA", torch.cuda.is_available())
EOF
