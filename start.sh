#!/bin/bash
# Start CSL GraphBuilder — frontend + backend
# Usage: ./start.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Kill any existing processes on our ports
echo "Cleaning up existing processes..."
lsof -ti:3000,8000 | xargs kill -9 2>/dev/null || true

# Activate Python virtualenv
source "$PROJECT_DIR/.venv/bin/activate"

# Start backend (FastAPI)
echo "Starting backend on http://localhost:8000 ..."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend (Next.js)
echo "Starting frontend on http://localhost:3000 ..."
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

cd "$PROJECT_DIR"

# Trap Ctrl+C to kill both
trap "echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo ""
echo "=== CSL GraphBuilder ==="
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Press Ctrl+C to stop both."
echo "========================"
echo ""

wait
