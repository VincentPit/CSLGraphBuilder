#!/bin/bash
# Start CSL GraphBuilder on offset ports so it doesn't conflict with the
# myquant stack already bound to 3000/8000.
#
#   Frontend → http://localhost:3010   (Next.js dev)
#   Backend  → http://localhost:8010   (FastAPI / uvicorn --reload)
#   Neo4j    → already running in Docker on 7474/7687
#
# Usage: ./start-local.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

FRONTEND_PORT=3010
BACKEND_PORT=8010

# Env overrides for the local dev profile.
#  * NEO4J_PASSWORD: don't force a value here — let the Python app load it
#    from .env via python-dotenv. Forcing one in the shell broke the switch
#    between the docker container (NEO4J_AUTH=neo4j/password) and the brew
#    install (auth=neo4j/changeme), since the chosen value rarely matched
#    whichever Neo4j was actually running.
#  * CORS_ORIGINS: must list the frontend origin explicitly. Wildcard "*"
#    combined with allow_credentials=True is rejected by browsers, so the API
#    must echo a concrete origin header. Add 3000 too for anyone running the
#    legacy script.
export CORS_ORIGINS="http://localhost:$FRONTEND_PORT,http://localhost:3000"

# Kill only processes listening on OUR ports — never touch 3000/8000 since
# those belong to another stack on this machine.
echo "Cleaning up any prior GraphBuilder dev servers on $FRONTEND_PORT / $BACKEND_PORT…"
lsof -ti:$FRONTEND_PORT,$BACKEND_PORT -sTCP:LISTEN 2>/dev/null | xargs kill -9 2>/dev/null || true

# Activate Python virtualenv
if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
  echo "ERROR: Python venv not found at $PROJECT_DIR/.venv"
  echo "       Create one with: python -m venv .venv && pip install -r requirements.txt"
  exit 1
fi
# shellcheck source=/dev/null
source "$PROJECT_DIR/.venv/bin/activate"

# Start backend (FastAPI). Inherits NEO4J_* / LLM_API_KEY from .env via dotenv.
echo "Starting backend on http://localhost:$BACKEND_PORT …"
uvicorn api.main:app --reload --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!

# Start frontend (Next.js). Tell it where the API actually lives —
# NEXT_PUBLIC_API_URL is read by frontend/lib/api.ts at build/dev time.
echo "Starting frontend on http://localhost:$FRONTEND_PORT …"
cd "$PROJECT_DIR/frontend"
NEXT_PUBLIC_API_URL="http://localhost:$BACKEND_PORT" \
  npm run dev -- --port $FRONTEND_PORT &
FRONTEND_PID=$!

cd "$PROJECT_DIR"

# Trap Ctrl+C / SIGTERM to kill both children.
trap "echo 'Shutting down…'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

cat <<EOF

=== CSL GraphBuilder (local-offset ports) ===
  Frontend: http://localhost:$FRONTEND_PORT
  Backend:  http://localhost:$BACKEND_PORT
  API Docs: http://localhost:$BACKEND_PORT/docs
  Neo4j:    http://localhost:7474   (already in Docker)
  Press Ctrl+C to stop both.
=============================================

EOF

wait
