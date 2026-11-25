#!/bin/bash
# Log channel test: runs test_cupti_prof (mock CUPTI), attaches bpftrace to
# the log and error probes after parcagpu has started, and checks that the
# records logged before the attach are replayed to it. No GPU needed.
#
# Build identity:  the library's .note.parcagpu ELF note carries the build
#                  info, the same string as the build record.
# Phase replay:    warnings (bad environment) and the startup record reach a
#                  consumer that attached after them, each once, on the log
#                  probe and (warnings) on the error probe; nothing goes to
#                  stderr without PARCAGPU_DEBUG.
# Phase subscribe: cuptiSubscribe fails (another CUPTI client): the error
#                  goes to stderr and is replayed, though parcagpu is inactive.
# Phase level:     PARCAGPU_LOG_LEVEL=warn keeps info records off the probe,
#                  except the build record.
# Phase pinned:    after more records than the replay ring holds, the build
#                  record is still replayed, first.
# Phase kernels:   a consumer attaching to cuda_correlation and kernel_stats
#                  (as parca-agent does) and detaching is logged as parcagpu
#                  starting and stopping recording every kernel.
#
# Prerequisites:
#   make local
#   bpftrace
#
# Usage:
#   sudo -E test/test-log.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIB="$ROOT/build-local/lib/libparcagpucupti.so"
TEST_BIN="$ROOT/build-local/bin/test_cupti_prof"

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

# bpftrace cuts strings at 64 bytes by default; messages are up to 256, but
# two strings that long overflow the BPF stack, and 160 covers the checks.
BT_PROG='config = { max_strlen = 160 }
usdt:'"$LIB"':parcagpu:log {
  printf("L%d t=%lu %s: %s\n", arg0, arg2, str(arg3), str(arg4));
}
usdt:'"$LIB"':parcagpu:error {
  printf("E code=%d %s: %s\n", (int32)arg0, str(arg2), str(arg1));
}'

# run_phase NAME [ENV...] — starts the workload with ENV, attaches once it
# has run a second, sets TEST_LOG / BT_LOG / ATTACH_NS.
run_phase() {
  local name="$1"; shift
  TEST_LOG="/tmp/parcagpu-log-$name-test.log"
  BT_LOG="/tmp/parcagpu-log-$name-bt.log"

  echo "=== Phase $name: $* ==="
  env LD_LIBRARY_PATH="$ROOT/build-local/lib:${LD_LIBRARY_PATH:-}" "$@" \
    "$TEST_BIN" "$LIB" --launch-rate=50 --duration=4 > "$TEST_LOG" 2>&1 &
  TEST_PID=$!
  while kill -0 "$TEST_PID" 2>/dev/null &&
        ! grep -q libparcagpucupti "/proc/$TEST_PID/maps" 2>/dev/null; do
    sleep 0.1
  done
  sleep 1 # well after parcagpu started and logged
  ATTACH_NS=$(date +%s%N)
  bpftrace -p "$TEST_PID" -e "$BT_PROG" > "$BT_LOG" 2>&1 || true
  wait "$TEST_PID" 2>/dev/null || true
  TEST_PID=""
  grep -E "^[LE]" "$BT_LOG" || true
}

count() { grep -cE "$1" "$2" || true; }

# --- Build identity ---
BUILD_INFO=$(strings "$LIB" | grep -m1 '^version=' || true)
echo "build info: $BUILD_INFO"
check_expr "ELF note .note.parcagpu present" \
  "readelf -n '$LIB' | grep -q 'Displaying notes found in: .note.parcagpu'"
check_expr "build info has version, sha, dirty, build_type, cuda_headers, cupti_api" \
  "echo '$BUILD_INFO' | grep -qE '^version=[^ ]+ sha=[^ ]+ dirty=(0|1|unknown) build_type=[^ ]+ cuda_headers=[^ ]+ cupti_api=[^ ]+$'"
if HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null); then
  check_expr "build info sha is HEAD" \
    "echo '$BUILD_INFO' | grep -q ' sha=$HEAD_SHA '"
fi
BUILD_VERSION=${BUILD_INFO%% *}; BUILD_VERSION=${BUILD_VERSION#version=}
# The build record as the log probe shows it.
BUILD_RE="^L2 t=[0-9]+ build: ${BUILD_INFO//+/\\+}\$"

# --- Replay ---
run_phase replay PARCAGPU_BOGUS=1 PARCAGPU_RATE_LIMIT=abc
check_expr "unknown variable warning replayed once on log" \
  "[ $(count "^L3 .* env_config: unrecognized env var 'PARCAGPU_BOGUS'" "$BT_LOG") -eq 1 ]"
check_expr "invalid value warning replayed once on log" \
  "[ $(count "^L3 .* env_config: PARCAGPU_RATE_LIMIT=abc invalid" "$BT_LOG") -eq 1 ]"
check_expr "startup record replayed once on log" \
  "[ $(count "^L2 .* cupti: parcagpu .* active: " "$BT_LOG") -eq 1 ]"
check_expr "build record replayed once on log" \
  "[ $(count "$BUILD_RE" "$BT_LOG") -eq 1 ]"
check_expr "build record is the first record" \
  "grep -E '^[LE]' '$BT_LOG' | head -1 | grep -qE '$BUILD_RE'"
check_expr "build record not on error" "[ $(count "^E .*: version=" "$BT_LOG") -eq 0 ]"
check_expr "startup record carries the version" \
  "[ $(count "^L2 .* cupti: parcagpu ${BUILD_VERSION//+/\\+} active: " "$BT_LOG") -eq 1 ]"
check_expr "startup record says kernel activity is on" \
  "[ $(count "^L2 .* active: .*, kernel activity on, " "$BT_LOG") -eq 1 ]"
check_expr "warnings replayed once on error" \
  "[ $(count "^E code=0 env_config: " "$BT_LOG") -eq 2 ]"
check_expr "info not on error" "[ $(count "^E .* active: " "$BT_LOG") -eq 0 ]"
OLDEST=$(grep -oE "^L[0-9] t=[0-9]+" "$BT_LOG" | cut -d= -f2 | sort -n | head -1)
check_expr "replayed records keep their time (before the attach)" \
  "[ -n '$OLDEST' ] && [ ${OLDEST:-0} -lt $ATTACH_NS ]"
check_expr "nothing on stderr without PARCAGPU_DEBUG" \
  "[ $(count "\[PARCAGPU\]" "$TEST_LOG") -eq 0 ]"

# --- Subscribe failure (39 = CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) ---
run_phase subscribe MOCK_CUPTI_SUBSCRIBE_RESULT=39
check_expr "subscribe failure on stderr" \
  "[ $(count "\[PARCAGPU\] ERROR cupti: parcagpu .* inactive: cuptiSubscribe failed" "$TEST_LOG") -eq 1 ]"
check_expr "subscribe failure replayed on log" \
  "[ $(count "^L4 .* cupti: parcagpu .* inactive: cuptiSubscribe failed: .* another CUPTI client" "$BT_LOG") -eq 1 ]"
check_expr "subscribe failure replayed on error" \
  "[ $(count "^E code=39 cupti: parcagpu .* inactive: cuptiSubscribe failed" "$BT_LOG") -eq 1 ]"

# --- Threshold ---
run_phase level PARCAGPU_LOG_LEVEL=warn PARCAGPU_BOGUS=1
check_expr "warning on log at warn" \
  "[ $(count "^L3 .* unrecognized env var" "$BT_LOG") -eq 1 ]"
check_expr "no info on log at warn but the build record" \
  "[ $(grep -E '^L[012] ' '$BT_LOG' | grep -cvE '$BUILD_RE' || true) -eq 0 ]"
check_expr "build record on log at warn" \
  "[ $(count "$BUILD_RE" "$BT_LOG") -eq 1 ]"

# --- Pinned: 80 warnings overflow the 64-record ring ---
BOGUS=()
for i in $(seq 1 80); do BOGUS+=("PARCAGPU_BOGUS_$i=1"); done
echo "=== Phase pinned: PARCAGPU_BOGUS_{1..80}=1 ==="
run_phase pinned "${BOGUS[@]}" > /dev/null
check_expr "ring overflowed (fewer than 80 warnings replayed)" \
  "[ $(count "^L3 .* unrecognized env var" "$BT_LOG") -lt 80 ]"
check_expr "build record still replayed once after the ring wrapped" \
  "[ $(count "$BUILD_RE" "$BT_LOG") -eq 1 ]"
check_expr "build record first after the ring wrapped" \
  "grep -E '^L' '$BT_LOG' | head -1 | grep -qE '$BUILD_RE'"

# --- Recording every kernel: a second consumer attaches, then detaches ---
TEST_LOG="/tmp/parcagpu-log-kernels-test.log"
BT_LOG="/tmp/parcagpu-log-kernels-bt.log"
echo "=== Phase kernels ==="
env LD_LIBRARY_PATH="$ROOT/build-local/lib:${LD_LIBRARY_PATH:-}" \
  "$TEST_BIN" "$LIB" --launch-rate=50 --duration=7 > "$TEST_LOG" 2>&1 &
TEST_PID=$!
while kill -0 "$TEST_PID" 2>/dev/null &&
      ! grep -q libparcagpucupti "/proc/$TEST_PID/maps" 2>/dev/null; do
  sleep 0.1
done
sleep 1
bpftrace -p "$TEST_PID" -e "$BT_PROG" > "$BT_LOG" 2>&1 &
LOG_BT=$!
sleep 1.5 # the log consumer's replay is done
timeout 2 bpftrace -p "$TEST_PID" -e 'usdt:'"$LIB"':parcagpu:cuda_correlation,
  usdt:'"$LIB"':parcagpu:kernel_stats { @n = count(); }' > /dev/null 2>&1 || true
wait "$TEST_PID" 2>/dev/null || true
TEST_PID=""
wait "$LOG_BT" 2>/dev/null || true
grep -E "^L" "$BT_LOG" | grep "every kernel" || true
check_expr "start of recording every kernel logged once" \
  "[ $(count "^L2 .* kernel_stats: recording every kernel: kernel_stats consumer attached" "$BT_LOG") -eq 1 ]"
check_expr "stop of recording every kernel logged once" \
  "[ $(count "^L2 .* kernel_stats: stopped recording every kernel: .* detached" "$BT_LOG") -eq 1 ]"
check_expr "start before stop" \
  "grep -E 'every kernel' '$BT_LOG' | head -1 | grep -q 'recording every kernel: '"

if $PASS; then
  echo; echo "=== ALL CHECKS PASSED ==="
else
  echo; echo "=== SOME CHECKS FAILED ===" >&2
  exit 1
fi
