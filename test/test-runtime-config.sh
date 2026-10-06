#!/bin/bash
# Runtime config and profiling gate against mock CUPTI. No GPU needed; the
# correlation checks attach bpftrace to enable the probes and are skipped
# without passwordless sudo.
#
#   1. PARCAGPU_ENABLED=0: no PC sampling windows open.
#   2. PARCAGPU_DUTY_CYCLE=0.5, PARCAGPU_DUTY_PERIOD=1: windows open, and
#      correlation probes fire, only in the first half of each wall-clock
#      second; profiling turns off and on repeatedly.
#   3. PARCAGPU_CONFIG_FILE, rewritten while the process runs:
#        ENABLED=0 (+ a bogus key, an env-only key) -> off, both warned about
#        ENABLED=1, PC_SAMPLING_RATE=0               -> on, PC sampling paused
#        ENABLED=1                                   -> PC sampling resumes
#      No window may open before the last rewrite.
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
export PARCAGPU_PC_SAMPLING_RATE=1000 PARCAGPU_DEBUG=1

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
check() { # check <name> <condition-result 0/1> <detail>
  if [ "$2" = 1 ]; then echo "ok   $1"; else echo "FAIL $1: $3"; fail=1; fi
}
starts() { sed -n 's/.*pc_sampling summary: starts=\([0-9]*\).*/\1/p' "$1"; }
# Fractional seconds of the debug-log timestamps on lines matching $2.
phases() { grep "$2" "$1" | sed -n 's/^\[[0-9]*\.\([0-9]\{3\}\).*/\1/p'; }
harness() { "$TEST_BIN" "$LIB" --threads=2 --num-gpus=2 --launch-rate=500 "$@"; }

# Correlation probes only fire with a consumer attached; bpftrace provides one.
have_bpftrace=0
if command -v bpftrace >/dev/null && sudo -n true 2>/dev/null; then
  have_bpftrace=1
fi
attach() { # attach <pid>: enable the cuda_correlation probe until the pid exits
  [ "$have_bpftrace" = 1 ] || return 0
  sudo -n bpftrace -p "$1" -e \
    "usdt:$LIB:parcagpu:cuda_correlation { @n = count(); }" >/dev/null 2>&1 &
}

# 1. Disabled from the env.
PARCAGPU_ENABLED=0 harness --duration=2 >"$TMP/1.log" 2>&1
n="$(starts "$TMP/1.log")"
check "ENABLED=0: no PC windows" "$([ "$n" = 0 ] && echo 1 || echo 0)" "starts=$n"

# 2. Duty cycle.
PARCAGPU_DUTY_CYCLE=0.5 PARCAGPU_DUTY_PERIOD=1 harness --duration=6 \
  >"$TMP/2.log" 2>&1 &
pid=$!
attach "$pid"
wait "$pid"
n="$(starts "$TMP/2.log")"
check "duty: PC windows open" "$([ "${n:-0}" -gt 0 ] && echo 1 || echo 0)" "starts=$n"
offs="$(grep -c 'profiling off' "$TMP/2.log" || true)"
check "duty: turns off repeatedly" "$([ "$offs" -ge 4 ] && echo 1 || echo 0)" "$offs offs"
# On for [0, 500) ms of each second; allow a few ms for the thread to flip.
late="$(phases "$TMP/2.log" 'PC sampling started' | awk '$1 >= 505' | wc -l)"
check "duty: windows only in on-phase" "$([ "$late" = 0 ] && echo 1 || echo 0)" \
  "$late started after 505ms"
if [ "$have_bpftrace" = 1 ]; then
  on="$(phases "$TMP/2.log" 'Inserted correlationId' | awk '$1 < 505' | wc -l)"
  off="$(phases "$TMP/2.log" 'Inserted correlationId' | awk '$1 >= 505' | wc -l)"
  check "duty: correlations only in on-phase" \
    "$([ "$on" -gt 0 ] && [ "$off" = 0 ] && echo 1 || echo 0)" "on=$on off=$off"
else
  echo "skip duty: correlations (needs bpftrace and passwordless sudo)"
fi

# 3. Config file, rewritten (atomically, like a ConfigMap) while running.
cfg="$TMP/parcagpu.conf"
put() { printf "$1" >"$cfg.tmp" && mv "$cfg.tmp" "$cfg"; }
put '# test\nENABLED=0\nBOGUS=1\nSAMPLING_FACTOR=10\n'
PARCAGPU_CONFIG_FILE="$cfg" harness --duration=6 >"$TMP/3.log" 2>&1 &
pid=$!
sleep 1.5; put 'PARCAGPU_ENABLED=1\nPC_SAMPLING_RATE=0\n'
sleep 1.5; resumed="$(date +%s.%N)"; put 'ENABLED=1\n'
wait "$pid"
for w in 'BOGUS=1 is not a known setting' \
         'SAMPLING_FACTOR=10 can only be set in the environment'; do
  check "file: warns '$w'" "$(grep -qF "$w" "$TMP/3.log" && echo 1 || echo 0)" missing
done
check "file: starts off" \
  "$(grep -m1 -E 'profiling (on|off)' "$TMP/3.log" | grep -q off && echo 1 || echo 0)" \
  "first transition isn't off"
check "file: turned on" "$(grep -q 'profiling on' "$TMP/3.log" && echo 1 || echo 0)" missing
first="$(grep -m1 'PC sampling started' "$TMP/3.log" | sed -n 's/^\[\([0-9.]*\)\].*/\1/p')"
check "file: PC windows resume" "$([ -n "$first" ] && echo 1 || echo 0)" "no windows"
if [ -n "$first" ]; then
  # The file is re-read once a second, so the first window may lag the write.
  check "file: no PC windows before resume" \
    "$(awk -v f="$first" -v r="$resumed" 'BEGIN { print (f >= r) ? 1 : 0 }')" \
    "first window at $first, resumed at $resumed"
fi

[ "$fail" = 0 ] && echo "PASS" || { echo "FAILED (logs in $TMP)"; trap - EXIT; exit 1; }
