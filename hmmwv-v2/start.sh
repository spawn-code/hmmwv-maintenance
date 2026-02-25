#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  MechAssist v2 — start.sh
#  Starts both the FastAPI backend (port 8000) and Vite dev server
#  (port 5173) in the background, then waits.
# ─────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# ── Locate Python (prefer the worktree venv) ─────────────────────
VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
  PYTHON="$VENV_PYTHON"
elif command -v python3 &>/dev/null; then
  PYTHON="$(command -v python3)"
else
  echo "ERROR: No Python found. Activate a virtual environment first."
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MechAssist — HMMWV Diagnostics v2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Python  : $PYTHON"
echo "  Backend : http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""

# ── Backend ──────────────────────────────────────────────────────
echo "[1/2] Starting FastAPI backend…"
cd "$BACKEND_DIR"
"$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# ── Frontend ─────────────────────────────────────────────────────
echo "[2/2] Starting Vite dev server…"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Both servers running. Press Ctrl-C to stop."
echo ""

# ── Cleanup on exit ──────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down…"
  kill "$BACKEND_PID"  2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

wait
