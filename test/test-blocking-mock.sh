#!/bin/bash
# Mock cuda_blocking_call test: runs test_cupti_prof (mock CUPTI) with fake
# stream syncs (--sync-us) and checks the cuda_blocking_call USDT probe.
# No GPU needed.
#
# Phase 1 (default threshold, 10 us): every 200 us sync is reported with
#   weight == duration; runtime syncs are reported once (cbid 131), not
#   again as their nested driver call; pure driver syncs report cbid -126.
# Phase 2 (PARCAGPU_BLOCKING_THRESHOLD_US=1000): 200 us syncs are below T, so
#   only some are reported, each with weight == T.
#
# Prerequisites:
#   make local
#   bpftrace
#
# Usage:
#   sudo -E test/test-blocking-mock.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIB="$ROOT/build-local/lib/libparcagpucupti.so"
TEST_BIN="$ROOT/build-local/bin/test_cupti_prof"
SYNC_US=200

for f in "$LIB" "$TEST_BIN"; do
  [ -e "$f" ] || { echo "error: $f not found" >&2; exit 1; }
done
command -v bpftrace >/dev/null || { echo "error: bpftrace not found" >&2; exit 1; }

cleanup() {
  [ -n "${TEST_PID:-}" ] && kill "$TEST_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

PASS=true
check_expr() {
  if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1" >&2; PASS=false; fi
}

# Per signed cbid: events, min duration, and weight mismatches against
# EXPECTED_WEIGHT (0 = expect weight == duration).
BT_PROG='usdt:'"$LIB"':parcagpu:cuda_blocking_call {
  @events[(int32)arg2] = count();
  @min_dur[(int32)arg2] = min(arg1 - arg0);
  @bad_weight[(int32)arg2] = sum(arg4 != ($1 == 0 ? arg1 - arg0 : $1) ? 1 : 0);
}'

# run_phase NAME THRESHOLD_US EXPECTED_WEIGHT_NS — sets TEST_LOG / BT_LOG.
run_phase() {
  local name="$1" threshold="$2" expected="$3"
  TEST_LOG="/tmp/parcagpu-blocking-mock-$name-test.log"
  BT_LOG="/tmp/parcagpu-blocking-mock-$name-bt.log"

  echo "=== Phase $name: sync-us=$SYNC_US threshold=${threshold:-default} ==="
  env LD_LIBRARY_PATH="$ROOT/build-local/lib:${LD_LIBRARY_PATH:-}" \
    ${threshold:+PARCAGPU_BLOCKING_THRESHOLD_US=$threshold} \
    "$TEST_BIN" "$LIB" --launch-rate=50 --sync-us=$SYNC_US --duration=5 \
    > "$TEST_LOG" 2>&1 &
  TEST_PID=$!
  while kill -0 "$TEST_PID" 2>/dev/null &&
        ! grep -q libparcagpucupti "/proc/$TEST_PID/maps" 2>/dev/null; do
    sleep 0.1
  done

  # bpftrace attaches after the workload starts (it needs the PID), so
  # early calls are missed; checks compare with <=. It exits with the target.
  bpftrace -p "$TEST_PID" -e "$BT_PROG" "$expected" > "$BT_LOG" 2>&1 || true
  wait "$TEST_PID" 2>/dev/null || true
  TEST_PID=""

  grep -E "^Sync calls:" "$TEST_LOG" || true
  grep -E "^@" "$BT_LOG" || true
}

calls() { grep -oE "$1=[0-9]+" "$TEST_LOG" | cut -d= -f2 || echo 0; }
val() { # val MAP CBID
  local v
  v=$(grep -E "^@$1\[$2\]: " "$BT_LOG" | awk '{ print $2 }' || true)
  echo "${v:-0}"
}

# --- Phase 1: default threshold ---
run_phase default "" 0
RT_CALLS=$(calls runtime); DRV_CALLS=$(calls driver)
RT_EV=$(val events 131); DRV_EV=$(val events -126)
OTHER=$(grep -E "^@events\[" "$BT_LOG" | grep -cvE "^@events\[(131|-126)\]" || true)

check_expr "runtime cudaStreamSynchronize reported (cbid 131)" "[ $RT_EV -gt 0 ]"
check_expr "driver cuStreamSynchronize reported (cbid -126)"  "[ $DRV_EV -gt 0 ]"
check_expr "runtime events <= runtime calls ($RT_EV <= $RT_CALLS)" "[ $RT_EV -le $RT_CALLS ]"
check_expr "nested driver call not double-reported ($DRV_EV <= $DRV_CALLS)" "[ $DRV_EV -le $DRV_CALLS ]"
check_expr "no unexpected cbids" "[ $OTHER -eq 0 ]"
check_expr "runtime durations >= ${SYNC_US}us" "[ $(val min_dur 131) -ge $((SYNC_US * 1000)) ]"
check_expr "driver durations >= ${SYNC_US}us"  "[ $(val min_dur -126) -ge $((SYNC_US * 1000)) ]"
check_expr "weight == duration above threshold" \
  "[ $(val bad_weight 131) -eq 0 ] && [ $(val bad_weight -126) -eq 0 ]"

# --- Phase 2: threshold above call duration ---
run_phase subthreshold 1000 1000000
CALLS=$(( $(calls runtime) + $(calls driver) ))
EV=$(( $(val events 131) + $(val events -126) ))

check_expr "sub-threshold calls sampled ($EV of $CALLS)" "[ $EV -gt 0 ] && [ $EV -lt $CALLS ]"
check_expr "sub-threshold weight == T" \
  "[ $(val bad_weight 131) -eq 0 ] && [ $(val bad_weight -126) -eq 0 ]"

if $PASS; then
  echo; echo "=== ALL CHECKS PASSED ==="
else
  echo; echo "=== SOME CHECKS FAILED ===" >&2
  exit 1
fi
