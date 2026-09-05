#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting COSMOCLIP (ISRO PS 26167) Prototype Stack"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure sample scenes are present
echo "[1/3] Verifying Sentinel-2 sample scenes..."
PYTHONPATH=. backend/venv/bin/python backend/scripts/generate_sample_scenes.py

# Start Backend
echo "[2/3] Starting FastAPI Backend on http://localhost:8000..."
PYTHONPATH=. backend/venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Trap cleanup
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT

# Wait for backend health
echo "Waiting for backend health check..."
for i in {1..15}; do
    if curl -s http://localhost:8000/api/health | grep -q "online"; then
        echo "Backend is healthy and online!"
        break
    fi
    sleep 1
done

# Start Frontend
echo "[3/3] Starting COSMOCLIP Cockpit UI on http://localhost:3000..."
cd frontend
npm run dev -- --port 3000 --host
