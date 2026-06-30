#!/usr/bin/env bash
# One-time setup for running finetune_mlx.ipynb in Cursor
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Using homework/.venv"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo "==> Registering Jupyter kernel: FineTune MLX"
python -m ipykernel install --user --name finetune-mlx --display-name "FineTune MLX"

echo "==> Verifying MLX"
python - <<'PY'
import mlx_lm
import mlx.core as mx
print("mlx-lm:", getattr(mlx_lm, "__version__", "installed"))
print("metal:", mx.metal.is_available())
PY

echo ""
echo "Done. In Cursor:"
echo "  1. Open homework/finetune_mlx.ipynb"
echo "  2. Kernel -> FineTune MLX"
echo "  3. Run All"
