#!/usr/bin/env bash
set -euo pipefail

backend_pid=""
nginx_pid=""

shutdown() {
    if [ -n "$backend_pid" ]; then
        kill -TERM "$backend_pid" 2>/dev/null || true
    fi
    if [ -n "$nginx_pid" ]; then
        kill -TERM "$nginx_pid" 2>/dev/null || true
    fi
    wait "$backend_pid" "$nginx_pid" 2>/dev/null || true
}

trap shutdown INT TERM

cd /app

mkdir -p .mitra/logs /run/nginx /tmp/huggingface /tmp/matplotlib

cat > .env <<EOF
LLM_TYPE=${LLM_TYPE:-}
LLM_API_KEY=${LLM_API_KEY:-}
LLM_MODEL=${LLM_MODEL:-}
LLM_GATEWAY_URL=${LLM_GATEWAY_URL:-}
LLM_CA_BUNDLE=${LLM_CA_BUNDLE:-}
EOF
chmod 600 .env

python -m uvicorn backend.main:create_app --factory --host 127.0.0.1 --port 8000 &
backend_pid=$!

nginx -g "daemon off;" &
nginx_pid=$!

wait -n "$backend_pid" "$nginx_pid"
exit_code=$?

shutdown
exit "$exit_code"
