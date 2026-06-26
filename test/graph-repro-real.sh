#!/usr/bin/env bash
# graph-repro-real.sh — real-GPU guard: runs graph_repro under the shim and
# checks the cuda_correlation USDT fires for driver cuGraphLaunch (cbid -514).
# Zero => FAIL (cbid not subscribed). Needs an NVIDIA GPU, nvcc, bpftrace, root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$ROOT/build-local/lib/libparcagpucupti.so"
SECS="${1:-15}"
BIN="$(mktemp -d)/graph_repro"
COUNT_LOG="$(mktemp)"

[ -f "$LIB" ] || { echo "error: $LIB not found — run 'make local' first" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "error: nvcc not found" >&2; exit 1; }
command -v bpftrace >/dev/null || { echo "error: bpftrace not found" >&2; exit 1; }

echo "=== Building reproducer ==="
nvcc -O2 -o "$BIN" "$ROOT/test/graph_repro.cu" -lcuda

echo "=== Launching reproducer under parcagpu shim ==="
LD_PRELOAD="$LIB" PARCAGPU_DEBUG=1 "$BIN" "$SECS" 2>"$(mktemp)" &
APP_PID=$!

# Wait for the shim to be mapped.
for _ in $(seq 1 50); do
  grep -q libparcagpucupti "/proc/$APP_PID/maps" 2>/dev/null && break
  sleep 0.1
done

echo "=== Attaching bpftrace to cuda_correlation USDT (driver cbid 514 => -514) ==="
# @driver: driver cuGraphLaunch (path under test); @runtime: positive cbid
bpftrace -p "$APP_PID" \
  -e "usdt:$LIB:parcagpu:cuda_correlation /arg1 == -514 || arg1 == -515/ { @driver = count(); }
      usdt:$LIB:parcagpu:cuda_correlation /arg1 >= 0/ { @runtime = count(); }" \
  >"$COUNT_LOG" 2>/dev/null &
BT_PID=$!

wait "$APP_PID" || true
sleep 1
kill "$BT_PID" 2>/dev/null || true
wait "$BT_PID" 2>/dev/null || true

echo "=== bpftrace results ==="
cat "$COUNT_LOG"

DRIVER=$(grep -oE "@driver: [0-9]+" "$COUNT_LOG" | grep -oE "[0-9]+" || echo 0)
echo
if [ "${DRIVER:-0}" -gt 0 ]; then
  echo "PASS: driver cuGraphLaunch produced $DRIVER correlation events"
  exit 0
else
  echo "FAIL: zero driver cuGraphLaunch correlation events — driver graph-launch" >&2
  echo "      cbid is not subscribed (CuptiProfiler::initialize)." >&2
  exit 1
fi
