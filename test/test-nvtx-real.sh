#!/bin/bash
# End-to-end NVTX test: real NVTX v3 clients load libparcagpucupti.so through
# NVTX_INJECTION64_PATH (the way torch, the Python nvtx package and
# TensorRT-LLM find it) and the BPF activity parser consumes the
# nvtx_event_batch / nvtx_resource_name USDT probes. Checks exact event
# counts, domain and registered-string resolution, thread naming, Ex
# attributes, multi-batch and stale-batch flushing, and that events from
# before a consumer attached are not captured. No GPU required.
#
# Cases:
#   c       test/nvtx_client.c via the stock nvtx3 headers (always)
#   python  test/nvtx_client.py via the Python nvtx package, if NVTX_PYTHON
#           names an interpreter that can `import nvtx`
#
# Prerequisites:
#   make local bpf-test
#
# Usage:
#   sudo -E test/test-nvtx-real.sh
#   sudo -E NVTX_PYTHON=/path/to/venv/bin/python test/test-nvtx-real.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIB="$ROOT/build-local/lib/libparcagpucupti.so"
CLIENT="$ROOT/build-local/bin/nvtx_client"
BPF="$ROOT/test/bpf/activity_parser"
WORK="$(mktemp -d /tmp/parcagpu-nvtx-real.XXXXXX)"

for f in "$LIB" "$CLIENT" "$BPF"; do
  if [ ! -e "$f" ]; then
    echo "error: $f not found (make local bpf-test)" >&2
    exit 1
  fi
done

cleanup() {
  [ -n "${CLIENT_PID:-}" ] && kill "$CLIENT_PID" 2>/dev/null || true
  [ -n "${BPF_PID:-}" ] && kill "$BPF_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

PASS=true

check_expr() {
  if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1" >&2; PASS=false; fi
}

# check_line LABEL PATTERN FILE: PASS if an exact line matching the extended
# regex PATTERN is in FILE.
check_line() {
  if grep -qxE "$2" "$3"; then
    echo "PASS: $1"
  else
    echo "FAIL: $1 (want /$2/)" >&2
    PASS=false
  fi
}

# run_case NAME CMD...: start the client, wait for it to load the library,
# attach the parser, then release the client and collect the parser summary
# in $WORK/NAME.bpf.log.
run_case() {
  local name="$1"
  shift
  local go="$WORK/$name.go"
  local client_log="$WORK/$name.client.log"
  local bpf_log="$WORK/$name.bpf.log"

  echo "=== [$name] starting client ==="
  # PARCAGPU_DEBUG would force capture on; this test relies on the probe
  # semaphores, as in production.
  env -u PARCAGPU_DEBUG NVTX_INJECTION64_PATH="$LIB" "$@" "$go" > "$client_log" 2>&1 &
  CLIENT_PID=$!

  while kill -0 "$CLIENT_PID" 2>/dev/null &&
        ! grep -q libparcagpucupti "/proc/$CLIENT_PID/maps" 2>/dev/null; do
    sleep 0.1
  done
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    echo "error: [$name] client exited before loading the library" >&2
    cat "$client_log" >&2
    exit 1
  fi

  "$BPF" -pid "$CLIENT_PID" -lib "$LIB" > "$bpf_log" 2>&1 &
  BPF_PID=$!
  # Wait for the probes to attach before releasing the client.
  for _ in $(seq 100); do
    grep -q "Waiting for events" "$bpf_log" && break
    sleep 0.1
  done
  touch "$go"

  wait "$CLIENT_PID" || { echo "error: [$name] client failed" >&2; cat "$client_log" >&2; exit 1; }
  CLIENT_PID=""
  wait "$BPF_PID" || true
  BPF_PID=""

  echo "--- [$name] client ---"
  cat "$client_log"
  echo "--- [$name] parser summary ---"
  grep '^nvtx' "$bpf_log" || { echo "(no nvtx summary)"; cat "$bpf_log"; }
  echo
}

# --- C client ---------------------------------------------------------------
run_case c "$CLIENT"
L="$WORK/c.bpf.log"
check_line "c: all 2301 events received, none dropped" \
  'nvtx: events=2301 bpf_records=2301 batches=[0-9]+ truncated=0 drops=0' "$L"
BATCHES=$(grep '^nvtx: ' "$L" | grep -oE 'batches=[0-9]+' | cut -d= -f2)
# 2251 main-thread events need >= 9 batches of 256; worker adds one; the
# spaced domain marks force extra stale-batch flushes.
check_expr "c: multiple batches incl. stale flushes (got ${BATCHES:-0})" "[ \"${BATCHES:-0}\" -ge 12 ]"
check_line "c: push count"  'nvtx kind push=1110' "$L"
check_line "c: pop count"   'nvtx kind pop=1110' "$L"
check_line "c: start count" 'nvtx kind start=8' "$L"
check_line "c: end count"   'nvtx kind end=8' "$L"
check_line "c: mark count"  'nvtx kind mark=65' "$L"
check_line "c: default domain events" 'nvtx domain id=0 name="" events=290' "$L"
check_line "c: domain named and attributed" 'nvtx domain id=1 name="parcagpu-test" events=2011' "$L"
check_line "c: registered string resolved" 'nvtx message "registered_step" count=1000' "$L"
check_line "c: ascii domain message" 'nvtx message "domain_mark" count=5' "$L"
check_line "c: worker thread marks" 'nvtx message "worker_mark" count=50' "$L"
check_line "c: two threads, Ex color/payload, ranges paired" \
  'nvtx threads=2 colored=10 payload_sum=70 open_ranges=0 bad_ends=0' "$L"
check_line "c: worker thread named" 'nvtx name kind=1 id=0x[0-9a-f]+ name="nvtx-worker"' "$L"
check_expr "c: pre-attach mark not captured" "! grep -q pre_attach \"$L\""

# --- Python nvtx package ------------------------------------------------------
if [ -n "${NVTX_PYTHON:-}" ] && "$NVTX_PYTHON" -c "import nvtx" 2>/dev/null; then
  run_case python "$NVTX_PYTHON" -I "$SCRIPT_DIR/nvtx_client.py"
  L="$WORK/python.bpf.log"
  check_line "python: all 210 events received" \
    'nvtx: events=210 bpf_records=210 batches=[0-9]+ truncated=0 drops=0' "$L"
  check_line "python: push count" 'nvtx kind push=100' "$L"
  check_line "python: pop count"  'nvtx kind pop=100' "$L"
  check_line "python: mark count" 'nvtx kind mark=10' "$L"
  check_line "python: domain named and attributed" \
    'nvtx domain id=[0-9]+ name="parcagpu-py" events=210' "$L"
  check_line "python: decorator message" 'nvtx message "py_step" count=100' "$L"
else
  echo "SKIP: python case (set NVTX_PYTHON to an interpreter with the nvtx package)"
fi

echo
echo "logs: $WORK"
if $PASS; then
  echo "=== ALL CHECKS PASSED ==="
else
  echo "=== SOME CHECKS FAILED ===" >&2
  exit 1
fi
