#!/bin/bash
# SecureMatAgent Docker entrypoint
# Starts FastAPI (background) + Streamlit (foreground).
# Proper signal propagation: SIGTERM kills both child processes.

set -e

FASTAPI_PORT=${API_PORT:-8000}
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}

echo "=== SecureMatAgent starting ==="
echo "FastAPI   → port $FASTAPI_PORT"
echo "Streamlit → port $STREAMLIT_PORT"
echo "Ollama    → ${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
echo "Qdrant    → ${QDRANT_HOST:-qdrant}:${QDRANT_PORT:-6333}"

# Trap SIGTERM/SIGINT and forward to children
_term() {
    echo "Caught SIGTERM — stopping children…"
    kill -TERM "$fastapi_pid" 2>/dev/null || true
    kill -TERM "$streamlit_pid" 2>/dev/null || true
    wait "$fastapi_pid" "$streamlit_pid"
    echo "All processes stopped."
}
trap _term SIGTERM SIGINT

# Start FastAPI in background
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$FASTAPI_PORT" \
    --workers 1 \
    --log-level info &
fastapi_pid=$!

# Start Streamlit in background
streamlit run ui/streamlit_app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false &
streamlit_pid=$!

echo "FastAPI PID=$fastapi_pid  Streamlit PID=$streamlit_pid"

# Wait for either process to exit
wait -n "$fastapi_pid" "$streamlit_pid"
EXIT_CODE=$?

echo "A child process exited ($EXIT_CODE) — stopping remaining processes…"
kill -TERM "$fastapi_pid" "$streamlit_pid" 2>/dev/null || true
wait

exit $EXIT_CODE
