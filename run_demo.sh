#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " 🛰️ Starting COSMOCLIP Multimodal AI Cockpit"
echo "=========================================================="

# 1. Check / Setup Python Virtual Environment
VENV_DIR="$SCRIPT_DIR/backend/venv"
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[1/4] Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"
    echo "Installing backend dependencies from requirements.txt..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
else
    # Verify core packages exist
    if ! "$VENV_DIR/bin/python" -c "import fastapi, uvicorn, scipy, PIL, pydantic" 2>/dev/null; then
        echo "[1/4] Installing missing Python dependencies..."
        "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
    else
        echo "[1/4] Python environment verified."
    fi
fi

# 2. Check / Setup Frontend Dependencies
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "[2/4] Installing frontend npm dependencies..."
    (cd "$SCRIPT_DIR/frontend" && npm install)
else
    echo "[2/4] Frontend dependencies verified."
fi

# 3. Ensure Sample Satellite Scenes & Cache Directories exist
echo "[3/4] Initializing satellite cache and sample scenes..."
mkdir -p "$SCRIPT_DIR/data/cache" "$SCRIPT_DIR/data/sample_scenes"
PYTHONPATH="$SCRIPT_DIR" "$VENV_DIR/bin/python" "$SCRIPT_DIR/backend/scripts/generate_sample_scenes.py"

# Free up Port 8000 if occupied by a previous zombie instance
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8000/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
    PID=$(lsof -ti:8000 2>/dev/null || true)
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null || true
    fi
fi

# 4. Start FastAPI Backend
echo "[4/4] Starting FastAPI backend on http://localhost:8000..."
PYTHONPATH="$SCRIPT_DIR" "$VENV_DIR/bin/uvicorn" backend.app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Trap cleanup to cleanly terminate backend on script exit
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT

# Wait for backend health check
echo "Waiting for backend health check..."
for i in {1..20}; do
    if curl -s http://localhost:8000/api/health | grep -q "online"; then
        echo "✅ Backend is healthy and online!"
        break
    fi
    sleep 1
done

# Start Frontend UI
echo "🚀 Launching COSMOCLIP Cockpit UI on http://localhost:3000..."
cd "$SCRIPT_DIR/frontend"
npm run dev -- --port 3000 --host
