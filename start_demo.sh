#!/usr/bin/env bash
# ==============================================================================
# PHANTOMFLOW — Unified Hackathon Demo Runner
# Starts:
#   1. Node.js Conversation & RailKit IRCTC Engine (Port 5001)
#   2. FastAPI Realtime WebSocket Server (Port 8000)
#   3. Midnight Signal Room Frontend Console (Port 5173)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "  Starting PHANTOMFLOW — Track 02 (The Midnight Phantom Hotline)"
echo "======================================================================"

# Clean up any stale processes on ports 5001, 8000, 5173
echo "[1/4] Checking ports..."
for PORT in 5001 8000 5173; do
  PID=$(lsof -ti :$PORT 2>/dev/null || true)
  if [ -n "$PID" ]; then
    echo "  Port $PORT in use by PID $PID. Killing existing process..."
    kill -9 $PID 2>/dev/null || true
  fi
done

# Start 1: Node.js Backend on 5001
echo "[2/4] Starting Node.js Backend (RailKit + Sarvam + Groq + MongoDB on Port 5001)..."
(cd backend && npm run dev) &
NODE_PID=$!

# Wait for 5001 to become healthy
echo "  Waiting for backend health check..."
for i in {1..15}; do
  if curl -s http://localhost:5001/health >/dev/null 2>&1; then
    echo "  Backend is HEALTHY on http://localhost:5001/health"
    break
  fi
  sleep 1
done

# Start 2: Python FastAPI Orchestrator on 8000
echo "[3/4] Starting FastAPI Realtime WebSocket Server (Port 8000)..."
PYTHONPATH=src "$SCRIPT_DIR/.venv/bin/python" -m uvicorn phantomflow.server:app --port 8000 --host 127.0.0.1 &
FASTAPI_PID=$!

# Wait for 8000 to become healthy
echo "  Waiting for orchestrator health check..."
for i in {1..15}; do
  if curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
    echo "  FastAPI Orchestrator is HEALTHY on http://localhost:8000/api/health"
    break
  fi
  sleep 1
done

# Start 3: Vite Frontend on 5173
echo "[4/4] Starting Midnight Signal Room Console (Port 5173)..."
(cd frontend && npm run dev) &
VITE_PID=$!

echo "======================================================================"
echo "  PHANTOMFLOW IS FULLY RUNNING!"
echo "  "
echo "  -> Signal Room UI Console : http://localhost:5173"
echo "  -> Realtime WebSocket Link: ws://localhost:8000/ws/voice"
echo "  -> Conversation & RailKit : http://localhost:5001/health"
echo "======================================================================"
echo "Press Ctrl+C to stop all services."

cleanup() {
  echo ""
  echo "Shutting down all services..."
  kill $NODE_PID $FASTAPI_PID $VITE_PID 2>/dev/null || true
  exit 0
}

trap cleanup INT TERM

wait
