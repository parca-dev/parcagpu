# Copyright 2026 The Parca Authors
# SPDX-License-Identifier: Apache-2.0
#
# Python nvtx package client for test-nvtx-real.sh, in the style of
# TensorRT-LLM: a domain-scoped decorator defined at import time (which
# creates the domain and registers the message before any consumer is
# attached), then a fixed workload once the go file appears.
#
# Usage: NVTX_INJECTION64_PATH=libparcagpucupti.so python3 nvtx_client.py <go-file>

import os
import sys
import time

import nvtx

STEPS = 100
MARKS = 10


@nvtx.annotate("py_step", color="green", domain="parcagpu-py")
def step():
    pass


def main():
    go_file = sys.argv[1]
    print(f"client: waiting for {go_file}", flush=True)
    deadline = time.monotonic() + 60
    while not os.path.exists(go_file):
        if time.monotonic() > deadline:
            sys.exit(f"client: timed out waiting for {go_file}")
        time.sleep(0.1)

    for _ in range(STEPS):
        step()
    for _ in range(MARKS):
        nvtx.mark("py_mark", domain="parcagpu-py")
    print(f"client: done, expected events={2 * STEPS + MARKS}", flush=True)


if __name__ == "__main__":
    main()
