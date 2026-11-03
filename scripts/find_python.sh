#!/usr/bin/env bash
# Print the path of a Python 3.13.5 interpreter (the version this project targets).
# Order: $PYTHON, then uv (installing 3.13.5 if needed), then python3.13.5 / python3.13 on PATH.
set -euo pipefail
WANT=3.13.5

check() { "$1" -c "import sys; sys.exit(0 if '%d.%d.%d' % sys.version_info[:3] == '$WANT' else 1)" 2>/dev/null; }

if [[ -n "${PYTHON:-}" ]]; then
  check "$PYTHON" && { echo "$PYTHON"; exit 0; }
  echo "PYTHON=$PYTHON is not Python $WANT" >&2; exit 1
fi
if command -v uv >/dev/null 2>&1; then
  uv python install "$WANT" >/dev/null 2>&1 || true
  P=$(uv python find "$WANT" 2>/dev/null || true)
  [[ -n "$P" ]] && check "$P" && { echo "$P"; exit 0; }
fi
for c in python3.13.5 python3.13 python3; do
  P=$(command -v "$c" 2>/dev/null || true)
  [[ -n "$P" ]] && check "$P" && { echo "$P"; exit 0; }
done
echo "Python $WANT not found. Install it (e.g. 'uv python install $WANT' or pyenv) or set PYTHON=/path/to/python3.13" >&2
exit 1
