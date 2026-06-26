#!/bin/bash
# Graph-launch variant of test-pc-mock.sh: drives test_cupti_prof with
# --graph-rate so half the launches are CUDA graph launches (one correlation
# ID fanning out into many kernel activity records with a nonzero graphId).
# Verifies graph kernels are profiled (emitted) rather than filtered, i.e. the
# PC-sampling rewrite did not break graph-launch profiling.
#
# Prereqs: make local bpf-test ; run with: sudo -E test/test-pc-mock-graph.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIB="$ROOT/build-local/lib/libparcagpucupti.so"
TEST_BIN="$ROOT/build-local/bin/test_cupti_prof"
BPF="$ROOT/test/bpf/activity_parser"
CUBIN="$ROOT/test/pc_sample_toy.cubin"
BPF_LOG="/tmp/parcagpu-graph-bpf.log"
TEST_LOG="/tmp/parcagpu-graph-test.log"

for f in "$LIB" "$TEST_BIN" "$BPF" "$CUBIN"; do
  [ -e "$f" ] || { echo "error: $f not found" >&2; exit 1; }
done

cleanup() {
  [ -n "${TEST_PID:-}" ] && kill "$TEST_PID" 2>/dev/null || true
  [ -n "${BPF_PID:-}" ] && kill "$BPF_PID" 2>/dev/null || true
  [ -n "${SEM_PID:-}" ] && kill "$SEM_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Starting test_cupti_prof (mock, graph-rate=25 of launch-rate=50) ==="
LD_LIBRARY_PATH="$ROOT/build-local/lib:${LD_LIBRARY_PATH:-}" \
  PARCAGPU_DEBUG=1 \
  PARCAGPU_PC_SAMPLING_RATE=10000 \
  MOCK_CUBIN_PATH="$CUBIN" \
  "$TEST_BIN" "$LIB" --launch-rate=50 --graph-rate=25 --duration=12 > "$TEST_LOG" 2>&1 &
TEST_PID=$!
echo "test_cupti_prof PID: $TEST_PID"

while kill -0 "$TEST_PID" 2>/dev/null &&
      ! grep -q libparcagpucupti "/proc/$TEST_PID/maps" 2>/dev/null; do
  sleep 0.1
done
if ! kill -0 "$TEST_PID" 2>/dev/null; then
  echo "error: test_cupti_prof exited before library loaded" >&2
  cat "$TEST_LOG" >&2; exit 1
fi

# The activity_parser does NOT attach the cuda_correlation probe, but
# parcagpu's allocBuffer + callback path are gated on its USDT semaphore
# (PARCAGPU_CUDA_CORRELATION_ENABLED). In production parca-agent attaches it;
# here we bump that semaphore with bpftrace so the activity path is live.
echo "=== Bumping cuda_correlation semaphore via bpftrace ==="
bpftrace -p "$TEST_PID" -e "usdt:$LIB:parcagpu:cuda_correlation { @n = count(); }
  usdt:$LIB:parcagpu:cuda_correlation /(int32)arg1 == -514 || (int32)arg1 == -515/ { @driver_graph = count(); }" \
  > /tmp/parcagpu-graph-sem.log 2>&1 &
SEM_PID=$!
sleep 2  # let bpftrace attach + set the semaphore before launches ramp

echo "=== Starting BPF activity parser ==="
"$BPF" -pid "$TEST_PID" -lib "$LIB" > "$BPF_LOG" 2>&1 &
BPF_PID=$!
echo "activity_parser PID: $BPF_PID"

wait "$TEST_PID" 2>/dev/null || true
TEST_PID=""
sleep 2
kill "$BPF_PID" 2>/dev/null || true
wait "$BPF_PID" 2>/dev/null || true
BPF_PID=""
kill "$SEM_PID" 2>/dev/null || true
wait "$SEM_PID" 2>/dev/null || true
SEM_PID=""
echo "=== bpftrace cuda_correlation hit count ==="
grep -E "@n:" /tmp/parcagpu-graph-sem.log || cat /tmp/parcagpu-graph-sem.log | tail -3 || true

echo
echo "=== graph map inserts (parcagpu debug, first 5) ==="
grep "into graph map" "$TEST_LOG" | head -5 || true
echo "=== graph map insert count ==="
grep -c "into graph map" "$TEST_LOG" || true
echo "=== filtered graph activities (graphId nonzero) count ==="
grep "Filtered kernel activity" "$TEST_LOG" | grep -vE "graphId=0\b" | grep -cE "graphId=[1-9]" || true
echo
echo "=== BPF: sample graph kernel events (graph= nonzero) ==="
grep -E "^kernel: " "$BPF_LOG" | grep -vE "graph=0\b" | grep -E "graph=[1-9]" | head -8 || true
echo "=== BPF: graph kernel-event count (graph= nonzero) ==="
grep -E "^kernel: " "$BPF_LOG" | grep -vE "graph=0\b" | grep -cE "graph=[1-9]" || true
echo "=== BPF: regular kernel-event count (graph=0) ==="
grep -cE "^kernel: .*graph=0\b" "$BPF_LOG" || true
echo "=== BPF summary line ==="
grep -E "kernels_found|events_received" "$BPF_LOG" | tail -2 || true
echo

# --- Checks ---
PASS=true
check() {
  if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1" >&2; PASS=false; fi
}
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
DRIVER_GRAPH=$(grep -oE "@driver_graph: [0-9]+" /tmp/parcagpu-graph-sem.log | grep -oE "[0-9]+" || echo 0)

check "graph launches inserted into graph map"         "[ \"${GRAPH_INSERTS:-0}\" -gt 0 ]"
check "graph kernel activities emitted (not filtered)"  "[ \"${GRAPH_EMITTED:-0}\" -gt 0 ]"
check "BPF consumer received kernel events"             "[ \"${BPF_RECEIVED:-0}\" -gt 0 ]"
check "driver cuGraphLaunch correlated (cbid subscribed)" "[ \"${DRIVER_GRAPH:-0}\" -gt 0 ]"

if $PASS; then echo; echo "=== GRAPH PROFILING VERIFIED ==="; else echo; echo "=== GRAPH PROFILING FAILED ===" >&2; exit 1; fi
