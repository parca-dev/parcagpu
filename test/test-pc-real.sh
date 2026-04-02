#!/bin/bash
# Smoke test: runs pc_sample_toy under parcagpu with BPF activity parser.
# Verifies stall reason map, PC samples, and cubin loading.
#
# Prerequisites:
#   make local bpf-test
#   nvcc -g -lineinfo -o microbenchmarks/pc_sample_toy microbenchmarks/pc_sample_toy.cu
#
# Usage:
#   sudo -E test/test-pc-real.sh          # default
#   sudo -E test/test-pc-real.sh -v       # verbose (print every event)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIB="$ROOT/build-local/lib/libparcagpucupti.so"
TOY="$ROOT/microbenchmarks/pc_sample_toy"
BPF="$ROOT/test/bpf/activity_parser"
BPF_LOG="/tmp/parcagpu-pc-test-bpf.log"
TOY_LOG="/tmp/parcagpu-pc-test-toy.log"
VERBOSE=""

for arg in "$@"; do
  case "$arg" in
    -v) VERBOSE="-v" ;;
  esac
done

# --- Preflight checks ---
for f in "$LIB" "$TOY" "$BPF"; do
  if [ ! -x "$f" ] && [ ! -f "$f" ]; then
    echo "error: $f not found" >&2
    exit 1
  fi
done

cleanup() {
  [ -n "${TOY_PID:-}" ] && kill "$TOY_PID" 2>/dev/null || true
  [ -n "${BPF_PID:-}" ] && kill "$BPF_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

# --- Launch the toy workload ---
echo "=== Starting pc_sample_toy ==="
PARCAGPU_DEBUG=1 PARCAGPU_SAMPLING_FACTOR=18 CUDA_INJECTION64_PATH="$LIB" "$TOY" 8 > "$TOY_LOG" 2>&1 &
TOY_PID=$!
echo "pc_sample_toy PID: $TOY_PID"

# Wait for library to be loaded into the process.
while kill -0 "$TOY_PID" 2>/dev/null &&
      ! grep -q libparcagpucupti "/proc/$TOY_PID/maps" 2>/dev/null; do
  sleep 0.1
done

if ! kill -0 "$TOY_PID" 2>/dev/null; then
  echo "error: pc_sample_toy exited before library loaded" >&2
  exit 1
fi

# --- Attach BPF parser ---
echo "=== Starting BPF activity parser ==="
"$BPF" -pid "$TOY_PID" -lib "$LIB" $VERBOSE > "$BPF_LOG" 2>&1 &
BPF_PID=$!
echo "activity_parser PID: $BPF_PID"

# --- Wait for workload to finish ---
wait "$TOY_PID" 2>/dev/null || true
TOY_PID=""
sleep 2

# --- Stop BPF parser ---
kill "$BPF_PID" 2>/dev/null || true
wait "$BPF_PID" 2>/dev/null || true
BPF_PID=""

# --- Results ---
echo
echo "=== Toy output (parcagpu debug) ==="
cat "$TOY_LOG"
echo
echo "=== BPF parser output ==="
cat "$BPF_LOG"
echo

# --- Checks ---
PASS=true

check() {
  local label="$1" pattern="$2" file="$3"
  if grep -q "$pattern" "$file"; then
    echo "PASS: $label"
  else
    echo "FAIL: $label" >&2
    PASS=false
  fi
}

check "modules loaded (parcagpu)" "Module 0x.*loaded" "$TOY_LOG"
check "stall reason map received" "stall reason map:" "$BPF_LOG"
check "PC samples contain stall reasons" "smsp__pcsamp" "$BPF_LOG"
check "cubins loaded (bpf)" "\[CUBIN\].*loaded" "$BPF_LOG"
check "PC sample events received" "pc_samples=[1-9]" "$BPF_LOG"

if $PASS; then
  echo
  echo "=== ALL CHECKS PASSED ==="
else
  echo
  echo "=== SOME CHECKS FAILED ===" >&2
  exit 1
fi
