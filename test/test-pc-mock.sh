#!/bin/bash
# Mock PC sampling test: runs test_cupti_prof (mock CUPTI) under parcagpu
# with BPF activity parser. Verifies stall reason map, PC samples, and
# cubin loading WITHOUT requiring a real GPU.
#
# Also drives a mix of CUDA graph launches (--graph-rate): half the launches
# fan one correlation ID out into many kernel activity records with a nonzero
# graphId. We verify those graph kernels are profiled (emitted) rather than
# filtered, and that the driver cuGraphLaunch cbid is subscribed — i.e. the
# PC-sampling rewrite did not break graph-launch profiling.
#
# Prerequisites:
#   make local bpf-test
#   bpftrace (used to bump the cuda_correlation USDT semaphore; see below)
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
SEM_LOG="/tmp/parcagpu-pc-mock-sem.log"
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
command -v bpftrace >/dev/null || { echo "error: bpftrace not found" >&2; exit 1; }

cleanup() {
  [ -n "${TEST_PID:-}" ] && kill "$TEST_PID" 2>/dev/null || true
  [ -n "${BPF_PID:-}" ] && kill "$BPF_PID" 2>/dev/null || true
  [ -n "${SEM_PID:-}" ] && kill "$SEM_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

# --- Launch mock workload ---
# Mock CUPTI/CUDA libs from build-local so proton's dynamic loader finds them
# instead of real libcupti.so / libcuda.so.
# High target rate keeps the controller's probability near 1 throughout the
# short mock run (otherwise it would converge below 1 once samples flow).
# --graph-rate=25 of --launch-rate=50: half the launches are CUDA graph
# launches (one correlation ID fanning out into many kernel records).
echo "=== Starting test_cupti_prof (mock, graph-rate=25 of launch-rate=50) ==="
LD_LIBRARY_PATH="$ROOT/build-local/lib:${LD_LIBRARY_PATH:-}" \
  PARCAGPU_DEBUG=1 \
  PARCAGPU_PC_SAMPLING_RATE=10000 \
  MOCK_CUBIN_PATH="$CUBIN" \
  "$TEST_BIN" "$LIB" --launch-rate=50 --graph-rate=25 --duration=15 > "$TEST_LOG" 2>&1 &
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

# The activity_parser does NOT attach the cuda_correlation probe, but
# parcagpu's allocBuffer + callback path are gated on its USDT semaphore
# (PARCAGPU_CUDA_CORRELATION_ENABLED). In production parca-agent attaches it;
# here we bump that semaphore with bpftrace so the activity path is live.
echo "=== Bumping cuda_correlation semaphore via bpftrace ==="
bpftrace -p "$TEST_PID" -e "usdt:$LIB:parcagpu:cuda_correlation { @n = count(); }
  usdt:$LIB:parcagpu:cuda_correlation /(int32)arg1 == -514 || (int32)arg1 == -515/ { @driver_graph = count(); }" \
  > "$SEM_LOG" 2>&1 &
SEM_PID=$!
sleep 2  # let bpftrace attach + set the semaphore before launches ramp

# --- Attach BPF parser ---
echo "=== Starting BPF activity parser ==="
"$BPF" -pid "$TEST_PID" -lib "$LIB" $VERBOSE > "$BPF_LOG" 2>&1 &
BPF_PID=$!
echo "activity_parser PID: $BPF_PID"

# --- Wait for workload to finish ---
wait "$TEST_PID" 2>/dev/null || true
TEST_PID=""
sleep 2

# --- Stop BPF parser + semaphore bumper ---
kill "$BPF_PID" 2>/dev/null || true
wait "$BPF_PID" 2>/dev/null || true
BPF_PID=""
kill "$SEM_PID" 2>/dev/null || true
wait "$SEM_PID" 2>/dev/null || true
SEM_PID=""

# --- Results ---
echo
echo "=== Mock test output (parcagpu debug) ==="
cat "$TEST_LOG"
echo
echo "=== BPF parser output ==="
cat "$BPF_LOG"
echo
echo "=== bpftrace cuda_correlation hit count ==="
grep -E "@n:" "$SEM_LOG" || tail -3 "$SEM_LOG" || true
echo "=== graph map insert count (parcagpu debug) ==="
grep -c "into graph map" "$TEST_LOG" || true
echo

# --- Checks ---
PASS=true

# grep-based check: PASS if PATTERN is present in FILE.
check() {
  local label="$1" pattern="$2" file="$3"
  if grep -q "$pattern" "$file"; then
    echo "PASS: $label"
  else
    echo "FAIL: $label" >&2
    PASS=false
  fi
}

# expression-based check: PASS if the shell test EXPR succeeds.
check_expr() {
  if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1" >&2; PASS=false; fi
}

# PC sampling + cubin + source correlation (no GPU).
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

# Graph-launch profiling.
GRAPH_INSERTS=$(grep -c "into graph map" "$TEST_LOG" || true)
# Authoritative signal: graph kernel activities that PASSED the correlation
# filter and were emitted (PARCAGPU_KERNEL_EXECUTED + activity_batch). The BPF
# per-kernel "kernel:" line is verbosity-gated (-v), so we count emits on the
# parcagpu side instead, and confirm the consumer received events.
GRAPH_EMITTED=$(grep "Kernel activity:" "$TEST_LOG" | grep -cE "graphId=[1-9]" || true)
BPF_RECEIVED=$(grep -oE "events_received=[0-9]+" "$BPF_LOG" | grep -oE "[0-9]+" | sort -n | tail -1 || echo 0)
# Driver cuGraphLaunch correlation events (signed cbid -514/-515). Zero means the
# driver graph-launch cbid wasn't subscribed; the runtime half emits regardless,
# so GRAPH_EMITTED>0 alone wouldn't catch it.
DRIVER_GRAPH=$(grep -oE "@driver_graph: [0-9]+" "$SEM_LOG" | grep -oE "[0-9]+" || echo 0)

check_expr "graph launches inserted into graph map"          "[ \"${GRAPH_INSERTS:-0}\" -gt 0 ]"
check_expr "graph kernel activities emitted (not filtered)"  "[ \"${GRAPH_EMITTED:-0}\" -gt 0 ]"
check_expr "BPF consumer received kernel events"             "[ \"${BPF_RECEIVED:-0}\" -gt 0 ]"
check_expr "driver cuGraphLaunch correlated (cbid subscribed)" "[ \"${DRIVER_GRAPH:-0}\" -gt 0 ]"

if $PASS; then
  echo
  echo "=== ALL CHECKS PASSED ==="
else
  echo
  echo "=== SOME CHECKS FAILED ===" >&2
  exit 1
fi
