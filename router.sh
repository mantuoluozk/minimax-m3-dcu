#!/usr/bin/env bash
set -euo pipefail
: "${PREFILL_URL:?Set PREFILL_URL}"
: "${DECODE_URL:?Set DECODE_URL}"
LOG_DIR="${LOG_DIR:-/tmp/minimax-m3-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/router-$(date +%Y%m%d-%H%M%S)-$$.log"
python3 -u -m sglang_router.launch_router \
  --pd-disaggregation --prefill "$PREFILL_URL" --decode "$DECODE_URL" \
  --host 0.0.0.0 --policy round_robin --port "${ROUTER_PORT:-40010}" \
  --prometheus-port "${PROMETHEUS_PORT:-3035}" \
  --health-check-endpoint /v1/models --health-check-interval-secs 60 \
  --health-check-timeout-secs 3600 --request-timeout-secs 36000 2>&1 | tee "$LOG_FILE"
