#!/bin/bash
# Mock PC sampling test: runs test_cupti_prof (mock CUPTI) under parcagpu
# with BPF activity parser. Verifies stall reason map, PC samples, and
# cubin loading WITHOUT requiring a real GPU.
#
# Prerequisites:
#   make local bpf-test
#
# Usage:
#   sudo -E test/test-pc-mock.sh          # default
#   sudo -E test/test-pc-mock.sh -v       # verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIB="$ROOT/build-local/lib/libparcagpucupti.so"
TEST_BIN="$ROOT/build-local/bin/test_cupti_prof"
BPF="$ROOT/test/bpf/activity_parser"
CUBIN="$ROOT/test/pc_sample_toy.cubin"
BPF_LOG="/tmp/parcagpu-pc-mock-bpf.log"
TEST_LOG="/tmp/parcagpu-pc-mock-test.log"
VERBOSE=""

for arg in "$@"; do
  case "$arg" in
    -v) VERBOSE="-v" ;;
  esac
done

# --- Preflight checks ---
for f in "$LIB" "$TEST_BIN" "$BPF" "$CUBIN"; do
  if [ ! -x "$f" ] && [ ! -f "$f" ]; then
    echo "error: $f not found" >&2
    exit 1
  fi
done

cleanup() {
  [ -n "${TEST_PID:-}" ] && kill "$TEST_PID" 2>/dev/null || true
  [ -n "${BPF_PID:-}" ] && kill "$BPF_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

# --- Launch mock workload ---
# Mock CUPTI/CUDA libs from build-local so proton's dynamic loader finds them
# instead of real libcupti.so / libcuda.so.
# Set probability=1 so every interval check triggers sampling.
echo "=== Starting test_cupti_prof (mock) ==="
LD_LIBRARY_PATH="$ROOT/build-local/lib:${LD_LIBRARY_PATH:-}" \
  PARCAGPU_DEBUG=1 \
  PARCAGPU_PC_SAMPLING_PROBABILITY=1 \
  PARCAGPU_PC_SAMPLING_INTERVAL=0.1 \
  MOCK_CUBIN_PATH="$CUBIN" \
  "$TEST_BIN" "$LIB" --launch-rate=50 --duration=15 > "$TEST_LOG" 2>&1 &
TEST_PID=$!
echo "test_cupti_prof PID: $TEST_PID"

# Wait for library to be loaded into the process.
while kill -0 "$TEST_PID" 2>/dev/null &&
      ! grep -q libparcagpucupti "/proc/$TEST_PID/maps" 2>/dev/null; do
  sleep 0.1
done

if ! kill -0 "$TEST_PID" 2>/dev/null; then
  echo "error: test_cupti_prof exited before library loaded" >&2
  cat "$TEST_LOG" >&2
  exit 1
fi

# --- Attach BPF parser ---
echo "=== Starting BPF activity parser ==="
"$BPF" -pid "$TEST_PID" -lib "$LIB" $VERBOSE > "$BPF_LOG" 2>&1 &
BPF_PID=$!
echo "activity_parser PID: $BPF_PID"

# --- Wait for workload to finish ---
wait "$TEST_PID" 2>/dev/null || true
TEST_PID=""
sleep 2

# --- Stop BPF parser ---
kill "$BPF_PID" 2>/dev/null || true
wait "$BPF_PID" 2>/dev/null || true
BPF_PID=""

# --- Results ---
echo
echo "=== Mock test output (parcagpu debug) ==="
cat "$TEST_LOG"
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

check "PC sampling initialized" "PC sampling initialized" "$TEST_LOG"
check "real cubin loaded (mock)" "Loaded cubin.*pc_sample_toy" "$TEST_LOG"
check "modules loaded (parcagpu)" "Module 0x.*loaded" "$TEST_LOG"
check "source correlation: shmem_bounce" "func=_Z12shmem_bounce.*pc_sample_toy.cu" "$TEST_LOG"
check "source correlation: hash_churn"  "func=_Z10hash_churn.*pc_sample_toy.cu"  "$TEST_LOG"
check "source correlation: trig_storm"  "func=_Z10trig_storm.*pc_sample_toy.cu"  "$TEST_LOG"
check "stall reason map received" "\[ 0\] smsp__pcsamp" "$BPF_LOG"
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
