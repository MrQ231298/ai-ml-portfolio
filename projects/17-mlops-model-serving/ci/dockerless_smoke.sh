#!/usr/bin/env bash
# Reproduces the Dockerfile steps without Docker (no Docker daemon on the build machine):
# fresh Python env with ONLY requirements-serve.txt, the same COPY layout, then the same
# uvicorn command, a /health check and one /predict call.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'kill ${PID:-0} 2>/dev/null || true; rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/srv/src"
cp "$HERE/requirements-serve.txt" "$STAGE/srv/"
cp "$HERE/src/drift.py" "$STAGE/srv/src/drift.py"
cp -r "$HERE/app" "$HERE/models" "$STAGE/srv/"
uv venv -q -p 3.13 "$STAGE/venv"
VIRTUAL_ENV="$STAGE/venv" uv pip install -q -r "$STAGE/srv/requirements-serve.txt"
cd "$STAGE/srv"
"$STAGE/venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8799 --workers 2 --log-level warning &
PID=$!
for i in $(seq 1 30); do curl -fsS localhost:8799/health && break; sleep 1; done
echo
curl -fsS -X POST localhost:8799/predict -H 'Content-Type: application/json' -d @"$HERE/tests/sample_request.json"
echo
echo "dockerless smoke test passed"
