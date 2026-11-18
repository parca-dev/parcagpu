#!/bin/bash
# End-to-end GPU memory profiling test: test/gpu_mem_client loads
# libparcagpucupti.so through CUDA_INJECTION64_PATH, and the BPF activity
# parser consumes both alloc/free providers:
#
#   parcagpu:gpu_mem_alloc/free   through the USDT argument specs
#   otel_gpu_memory:alloc/free    straight from the argument registers, with
#                                 an in-kernel (pid, ptr) map that drops frees
#                                 of unsampled pointers, the way the
#                                 opentelemetry-ebpf-profiler heap probe
#                                 (#1747) consumes otel_memory
#
# Checks exact counts, bytes, kinds and the in-use set with every allocation
# sampled; weighted totals under the default sampling interval; that both
# providers see the same sampling decisions; and that allocations made
# before a consumer attached are not reported.
#
# Modes:
#   mock  (default) gpu_mem_client against test/mock_cuda.c, which fires the
#         CUPTI callbacks itself. No GPU.
#   real  gpu_mem_client_real against the real driver. Needs a GPU.
#
# Prerequisites:
#   make local bpf-test
#
# Usage:
#   sudo -E test/test-gpu-mem-e2e.sh [mock|real]

set -euo pipefail

MODE="${1:-mock}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIB="$ROOT/build-local/lib/libparcagpucupti.so"
BPF="$ROOT/test/bpf/activity_parser"
case "$MODE" in
  mock)
    CLIENT="$ROOT/build-local/bin/gpu_mem_client"
    CLIENT_ENV=(LD_LIBRARY_PATH="$ROOT/build-local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
    ;;
  real)
    CLIENT="$ROOT/build-local/bin/gpu_mem_client_real"
    CLIENT_ENV=()
    ;;
  *)
    echo "usage: $0 [mock|real]" >&2
    exit 2
    ;;
esac
WORK="$(mktemp -d /tmp/parcagpu-gpu-mem.XXXXXX)"

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

# field FILE PREFIX KEY: value of KEY=... on the line starting with PREFIX.
field() {
  grep -E "^$2 " "$1" | grep -oE "(^| )$3=[0-9]+" | head -1 | cut -d= -f2
}

# run_case NAME WORKLOAD INTERVAL: start the client, wait for it to load the
# library, attach the parser, release the client, and collect the parser
# summary in $WORK/NAME.bpf.log. INTERVAL "" leaves the default.
run_case() {
  local name="$1" workload="$2" interval="$3"
  local go="$WORK/$name.go"
  local client_log="$WORK/$name.client.log"
  local bpf_log="$WORK/$name.bpf.log"
  local envs=("${CLIENT_ENV[@]}" CUDA_INJECTION64_PATH="$LIB")
  [ -n "$interval" ] && envs+=(PARCAGPU_MEM_SAMPLING_INTERVAL="$interval")

  echo "=== [$name] starting client ($MODE, $workload) ==="
  # PARCAGPU_DEBUG is unset: the test relies on the probe semaphores.
  env -u PARCAGPU_DEBUG "${envs[@]}" "$CLIENT" "$go" "$workload" \
    > "$client_log" 2>&1 &
  CLIENT_PID=$!

  while kill -0 "$CLIENT_PID" 2>/dev/null && ! grep -q '^ready' "$client_log"; do
    sleep 0.1
  done
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    echo "error: [$name] client exited before it was ready" >&2
    cat "$client_log" >&2
    exit 1
  fi

  "$BPF" -pid "$CLIENT_PID" -lib "$LIB" > "$bpf_log" 2>&1 &
  BPF_PID=$!
  for _ in $(seq 100); do
    grep -q "Waiting for events" "$bpf_log" && break
    sleep 0.1
  done
  touch "$go"

  wait "$CLIENT_PID" || { echo "error: [$name] client failed" >&2; cat "$client_log" >&2; exit 1; }
  CLIENT_PID=""
  wait "$BPF_PID" || true
  BPF_PID=""

  echo "--- [$name] parser summary ---"
  grep '^gpu_mem ' "$bpf_log" || { echo "(no gpu_mem summary)"; cat "$bpf_log"; }
  echo
}

# --- exact: every allocation sampled ----------------------------------------
run_case exact exact 0
L="$WORK/exact.bpf.log"
for src in parcagpu otel; do
  check_line "exact/$src: 16 allocs, 44 MiB, weight == size" \
    "gpu_mem $src allocs=16 bytes=46137344 weighted=46137344" "$L"
  check_line "exact/$src: 8 frees, all matched (pre-attach free not seen)" \
    "gpu_mem $src frees=8 unmatched_frees=0" "$L"
  check_line "exact/$src: in use 8 allocs, 22 MiB" \
    "gpu_mem $src inuse allocs=8 bytes=23068672 weighted=23068672" "$L"
done
check_line "exact: device allocs"  'gpu_mem parcagpu kind device=8' "$L"
check_line "exact: managed allocs" 'gpu_mem parcagpu kind managed=2' "$L"
check_line "exact: async allocs"   'gpu_mem parcagpu kind async=4' "$L"
check_line "exact: vmm maps"       'gpu_mem parcagpu kind vmm=2' "$L"
check_line "exact: one device"     'gpu_mem parcagpu devices=1' "$L"

# --- bulk: default interval (512 KiB), 2000 x 256 KiB ------------------------
run_case bulk bulk ""
L="$WORK/bulk.bpf.log"
TRUE_BYTES=$((2000 * 262144))
for src in parcagpu otel; do
  A=$(field "$L" "gpu_mem $src" allocs)
  W=$(field "$L" "gpu_mem $src" weighted)
  F=$(field "$L" "gpu_mem $src" frees)
  U=$(field "$L" "gpu_mem $src" unmatched_frees)
  echo "    $src: sampled=${A:-?} weighted=${W:-?} (true $TRUE_BYTES) frees=${F:-?} unmatched=${U:-?}"
  check_expr "bulk/$src: sampling thinned the stream" \
    "[ \"${A:-0}\" -gt 0 ] && [ \"${A:-0}\" -lt 2000 ]"
  # p = 1 - e^-0.5 per alloc; 15% is over 5 standard deviations.
  check_expr "bulk/$src: weighted bytes within 15% of the true total" \
    "awk -v w=\"${W:-0}\" -v t=$TRUE_BYTES 'BEGIN { r = w / t; exit !(r > 0.85 && r < 1.15) }'"
  check_expr "bulk/$src: every sampled alloc freed, other frees unmatched" \
    "[ \"${F:-x}\" = \"${A:-y}\" ] && [ \$(( ${F:-0} + ${U:-0} )) -eq 2000 ]"
  check_line "bulk/$src: nothing left in use" \
    "gpu_mem $src inuse allocs=0 bytes=0 weighted=0" "$L"
done
check_expr "bulk: both providers saw the same sampling decisions" \
  "[ \"\$(field \"$L\" 'gpu_mem parcagpu' weighted)\" = \"\$(field \"$L\" 'gpu_mem otel' weighted)\" ]"

echo
if $PASS; then
  echo "=== ALL CHECKS PASSED ==="
  rm -rf "$WORK"
else
  echo "=== SOME CHECKS FAILED (logs in $WORK) ==="
  exit 1
fi
