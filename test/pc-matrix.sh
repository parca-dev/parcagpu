#!/bin/bash
# Which PC sampling arrangements work across processes and devices, on this
# node's GPUs? Runs scripted pc_probe processes (no parcagpu) and prints, for
# each scenario, every process's samples per second per device and its CUPTI
# results. Needs PC sampling permission (root, or
# NVreg_RestrictProfilingToAdminUsers=0) and pc_probe:
#
#   nvcc -O2 -lineinfo -o test/pc_probe test/pc_probe.cu -lcuda -lcupti
#   test/pc-matrix.sh [scenario...]     # default: all that fit the GPU count
#
# The questions, and the scenarios that answer them:
#   one process, several GPUs at once ............................ multi_dev
#   several processes, one GPU each, at once (per-GPU sampler?) ... per_dev, staggered
#   does enabling on one GPU disturb sampling on another? ......... cross_enable
#   same GPU: does Stop hand the sampler over, or only Disable? ... stop_handover, disable_handover
#   same GPU: does a second process enabling disturb the first? ... second_enabler
#
# Results go to $OUT (default /tmp/pc-matrix), one JSON-lines file per process.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="${PC_PROBE:-$ROOT/test/pc_probe}"
OUT="${OUT:-/tmp/pc-matrix}"
NGPU="$(nvidia-smi -L | wc -l)"
[ -x "$PROBE" ] || { echo "build $PROBE first (see header)" >&2; exit 1; }

# run NAME SECS "devs|script" ...: one pc_probe process per argument.
run() {
  local name="$1" secs="$2"; shift 2
  local dir="$OUT/$name"; rm -rf "$dir"; mkdir -p "$dir"
  local i=0 pids=()
  for spec in "$@"; do
    "$PROBE" --devs="${spec%%|*}" --secs="$secs" --script="${spec#*|}" \
      > "$dir/p$i.jsonl" 2> "$dir/p$i.err" &
    pids+=($!); i=$((i + 1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  report "$name" "$@"
}

# One line per process and device: samples each second, then CUPTI results.
report() {
  local name="$1"; shift
  echo "=== $name"
  local i=0
  for spec in "$@"; do
    local f="$OUT/$name/p$i.jsonl"
    for dev in $(echo "${spec%%|*}" | tr , ' '); do
      local secs res errs
      secs=$(grep "\"dev\":$dev,.*\"samples\"" "$f" | sed 's/.*"samples":\([0-9]*\).*/\1/' | tr '\n' ' ')
      local rounds
      rounds=$(grep "\"dev\":$dev,.*\"rounds\"" "$f" | sed 's/.*"rounds":\([0-9]*\),"max_round_ms":\([0-9.]*\).*/\1(\2)/' | tr '\n' ' ')
      res=$(grep "\"dev\":$dev,.*\"action\"" "$f" | sed 's/.*"t":\([0-9.]*\).*"action":"\([a-z]*\)","res":"\([A-Z_]*\)".*/\2@\1=\3/' | tr '\n' ' ')
      errs=$(grep "\"dev\":$dev,.*\"getdata\"" "$f" | grep -v CUPTI_SUCCESS | sed 's/.*"t":\([0-9]*\).*"getdata":"\([A-Z_]*\)".*/\2@\1/' | sort -u | tr '\n' ' ')
      printf "  p%d dev%d  samples/s: %s\n          rounds/s(max ms): %s\n          %s%s\n" "$i" "$dev" "$secs" "$rounds" "$res" "${errs:+ getdata: $errs}"
    done
    [ -s "$OUT/$name/p$i.err" ] && sed "s/^/  p$i stderr: /" "$OUT/$name/p$i.err" | head -3
    i=$((i + 1))
  done
}

all_devs=$(seq -s, 0 $((NGPU - 1)))
per_dev_specs() { for d in $(seq 0 $((NGPU - 1))); do echo "$d|$1"; done; }

scenario() {
  case "$1" in
  multi_dev) # one process samples every GPU at once
    run multi_dev 6 "$all_devs|enable@0,start@0" ;;
  per_dev) # THE open question: one process per GPU, all sampling at once
    mapfile -t specs < <(per_dev_specs "enable@0,start@0")
    run per_dev 8 "${specs[@]}" ;;
  staggered) # process i enables+starts at second 2i: do later ones fail, or knock out earlier ones?
    local specs=() d
    for d in $(seq 0 $((NGPU - 1))); do specs+=("$d|enable@$((2 * d)),start@$((2 * d))"); done
    run staggered $((2 * NGPU + 4)) "${specs[@]}" ;;
  cross_enable) # GPU 0 sampling; another process only enables GPU 1 at 2s, starts at 4s
    run cross_enable 7 "0|enable@0,start@0" "1|enable@2,start@4" ;;
  stop_handover) # same GPU: p0 stops (no disable) at 3s; p1 enabled at 1s starts at 4s
    run stop_handover 8 "0|enable@0,start@0,stop@3" "0|enable@1,start@4" ;;
  disable_handover) # same GPU: p0 stops and disables at 3s; p1 enables and starts at 4s (parca-agent's hand-over)
    run disable_handover 8 "0|enable@0,start@0,stop@3,disable@3" "0|enable@4,start@4" ;;
  second_enabler) # same GPU: p0 sampling; p1 only enables at 2s: does p0 keep its samples?
    run second_enabler 6 "0|enable@0,start@0" "0|enable@2" ;;
  warmup_start) # what a window costs when the context stays enabled: 1s windows every 3s
    run warmup_start 11 "0|enable@0,start@1,stop@2,start@4,stop@5,start@7,stop@8" ;;
  warmup_enable) # ... and when each window enables afresh (parca-agent's grants)
    run warmup_enable 11 "0|enable@1,start@1,stop@2,disable@2,enable@4,start@4,stop@5,disable@5,enable@7,start@7,stop@8,disable@8" ;;
  *) echo "unknown scenario $1" >&2; exit 2 ;;
  esac
}

if [ $# -gt 0 ]; then
  for s in "$@"; do scenario "$s"; done
else
  scenario stop_handover; scenario disable_handover; scenario second_enabler
  if [ "$NGPU" -gt 1 ]; then
    scenario multi_dev; scenario per_dev; scenario staggered; scenario cross_enable
  fi
fi
