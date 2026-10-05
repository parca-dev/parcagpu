#!/bin/bash
# PC sampling context bookkeeping against mock CUPTI, which tracks start/stop
# per context and rejects starting a started context or stopping a stopped
# one (as real CUPTI does). No GPU or root needed.
#
#   1. 8 GPUs on one launcher thread: the thread switches contexts between
#      window open and close, so closing must stop the window's own context.
#      (The old code stopped the closing launch's context and stranded the
#      opened one in KERNEL_SERIALIZED sampling.)
#   2. 8 GPUs on 8 threads: windows on different contexts overlap; each must
#      be stopped exactly once, and several contexts may sample at once.
#   3. Context 3's sampler owned by another process (MOCK_PC_UNKNOWN_CTX):
#      GetData returns CUPTI_ERROR_UNKNOWN; we must stop and never restart it.
#
# Windows still open when the harness exits are expected (the mock never
# fires CONTEXT_DESTROY_STARTING), so still_started is bounded by threads.
#
# Prerequisites: make local
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB="$ROOT/build-local/lib/libparcagpucupti.so"
TEST_BIN="$ROOT/build-local/bin/test_cupti_prof"
for f in "$LIB" "$TEST_BIN"; do
  [ -f "$f" ] || { echo "error: $f not found (run 'make local')" >&2; exit 1; }
done
export LD_LIBRARY_PATH="$ROOT/build-local/lib:${LD_LIBRARY_PATH:-}"
export PARCAGPU_PC_SAMPLING_RATE=1000

fail=0
# run <name> <threads> <checks...> -- <test_cupti_prof args>
# Each check is "<field><op><value>" with op one of = <= >=.
run() {
  local name="$1" threads="$2"; shift 2
  local checks=()
  while [ "$1" != "--" ]; do checks+=("$1"); shift; done; shift
  local line
  line="$("$TEST_BIN" "$LIB" --threads="$threads" --launch-rate=2000 \
            --duration=3 "$@" 2>&1 | grep 'pc_sampling summary' || true)"
  if [ -z "$line" ]; then
    echo "FAIL $name: no pc_sampling summary"; fail=1; return
  fi
  local c field op want got ok
  for c in "${checks[@]}"; do
    [[ "$c" =~ ^([a-z_]+)(<=|>=|=)([0-9]+)$ ]] || { echo "bad check $c"; exit 2; }
    field="${BASH_REMATCH[1]}"; op="${BASH_REMATCH[2]}"; want="${BASH_REMATCH[3]}"
    got="$(sed -n "s/.* $field=\([0-9]*\).*/\1/p" <<<"$line")"
    case "$op" in
      =)  [ "$got" -eq "$want" ] && ok=1 || ok=0 ;;
      '<=') [ "$got" -le "$want" ] && ok=1 || ok=0 ;;
      '>=') [ "$got" -ge "$want" ] && ok=1 || ok=0 ;;
    esac
    if [ "$ok" = 0 ]; then
      echo "FAIL $name: $field=$got, want $op$want"; fail=1
    fi
  done
  echo "$name: ${line#*summary: }"
}

run "1 thread, 8 ctx " 1 starts\>=5 bad_starts=0 bad_stops=0 still_started\<=1 \
  -- --num-gpus=8
run "8 threads, 8 ctx" 8 starts\>=5 bad_starts=0 bad_stops=0 max_concurrent\>=2 still_started\<=8 \
  -- --num-gpus=8
MOCK_PC_UNKNOWN_CTX=3 \
run "ctx 3 unavailable" 8 bad_starts=0 bad_stops=0 unknown_ctx_starts\<=1 getdata_unknown\<=8 \
  -- --num-gpus=8

[ "$fail" = 0 ] && echo "PASS" || { echo "FAILED"; exit 1; }
