#!/bin/bash
# Drives test_cupti_prof at a production TRT-LLM 4-GPU decode peak: one process
# simulating 4 GPUs, each on its own launcher thread (its own thread_local token
# bucket, mirroring the 4 rank processes in prod).
#
# Rate derivation (peak ~740 graph launches/s, ~1760 total/s aggregate):
#   test_cupti_prof issues 5 launches/iteration, so actual = 5 * rate * num_gpus.
#     graph:  5 * 37 * 4 ~= 740/s
#     total:  5 * 88 * 4 ~= 1760/s
#
# Set PARCAGPU_GRAPH_RATE_LIMIT (and PARCAGPU_DEBUG=1) in the environment to
# observe throttling. Extra args are forwarded to test_cupti_prof (override the
# defaults). Attach a consumer (bpftrace/agent) separately to enable the probes.
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

exec env LD_LIBRARY_PATH="$ROOT/build-local/lib:${LD_LIBRARY_PATH:-}" \
  "$TEST_BIN" "$LIB" \
    --threads=4 --num-gpus=4 --launch-rate=88 --graph-rate=37 "$@"
