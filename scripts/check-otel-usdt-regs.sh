#!/usr/bin/env bash
# Copyright 2026 The Parca Authors
# SPDX-License-Identifier: Apache-2.0
#
# Check that the otel_gpu_memory USDT probes pass their arguments in the ABI
# argument registers. The opentelemetry-ebpf-profiler heap probe (#1747)
# reads them from there and ignores the SDT argument descriptors, so a
# compiler that moved an operand would make it read garbage.
#
# Usage: check-otel-usdt-regs.sh path/to/libparcagpucupti.so
set -euo pipefail

lib=${1:?usage: $0 <library>}
READELF=${READELF:-readelf}

machine=$("$READELF" -h "$lib" | awk -F: '/Machine:/ {gsub(/^[ \t]+/, "", $2); print $2}')
case "$machine" in
  *X86-64*)  want_alloc='^-?8@%rdi -?8@%rsi -?8@%rdx$'; want_free='^-?8@%rdi$' ;;
  *AArch64*) want_alloc='^-?8@x0 -?8@x1 -?8@x2$';      want_free='^-?8@x0$' ;;
  *) echo "check-otel-usdt-regs: unsupported machine '$machine'" >&2; exit 1 ;;
esac

# Print "name<TAB>arguments" for every otel_gpu_memory note.
notes=$("$READELF" -n "$lib" | awk '
  /Provider:/  { provider = $2 }
  /Name:/      { name = $2 }
  /Arguments:/ {
    if (provider == "otel_gpu_memory") {
      sub(/^[ \t]*Arguments:[ \t]*/, "")
      print name "\t" $0
    }
  }')

fail=0
for probe in alloc free; do
  want=$want_alloc
  [ "$probe" = free ] && want=$want_free
  lines=$(printf '%s\n' "$notes" | awk -F'\t' -v p="$probe" '$1 == p { print $2 }')
  count=$(printf '%s' "$lines" | grep -c . || true)
  if [ "$count" -ne 1 ]; then
    echo "check-otel-usdt-regs: expected 1 otel_gpu_memory:$probe site, found $count" >&2
    fail=1
    continue
  fi
  if ! printf '%s\n' "$lines" | grep -qE "$want"; then
    echo "check-otel-usdt-regs: otel_gpu_memory:$probe has arguments '$lines'," \
         "want /$want/" >&2
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "otel_gpu_memory USDT argument registers OK ($machine)"
