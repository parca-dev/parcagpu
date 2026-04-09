#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB="$SCRIPT_DIR/build-local/lib/libparcagpucupti.so"

if [ ! -f "$LIB" ]; then
    echo "error: $LIB not found" >&2
    exit 1
fi

export PARCAGPU_DEBUG=1
PARCAGPU_SAMPLING_FACTOR=18 CUDA_INJECTION64_PATH="$LIB" "$SCRIPT_DIR/microbenchmarks/pc_sample_toy" 4 &
TOY_PID=$!
trap "kill $TOY_PID 2>/dev/null; wait $TOY_PID 2>/dev/null" EXIT

# Wait for the injection library to be loaded (CUDA must initialize first)
while kill -0 "$TOY_PID" 2>/dev/null && ! grep -q libparcagpucupti "/proc/$TOY_PID/maps" 2>/dev/null; do
    sleep 0.1
done

exec bpftrace -p "$TOY_PID" "$SCRIPT_DIR/parcagpu.bt"
