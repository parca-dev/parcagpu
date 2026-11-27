# parcagpu

A CUDA injection library that exposes GPU activity through USDT probes for eBPF
consumers such as [parca-agent](https://github.com/parca-dev/parca-agent):
kernel executions and counts, PC samples with stall reasons, host time blocked
in CUDA calls, cubin loads, devices and NVTX ranges.

## Usage

```bash
export CUDA_INJECTION64_PATH=/path/to/libparcagpucupti.so
export NVTX_INJECTION64_PATH=/path/to/libparcagpucupti.so   # optional, for NVTX
./my_cuda_app
```

parcagpu is also an NVTX injection library (`InitializeInjectionNvtx2`). It
records ranges and marks from torch.cuda.nvtx, the Python `nvtx` package,
TensorRT-LLM, cuDNN and others, while a consumer is attached to
`nvtx_event_batch`. Each NVTX client loads its injection on its first NVTX call,
often at import time before CUDA initializes, and a client that finds none stays
a no-op. So set `NVTX_INJECTION64_PATH` in the launcher. parcagpu sets it at
`cuInit` if unset, which only covers clients that haven't called NVTX yet. The
slot holds one library, so this can't be combined with Nsight Systems' NVTX
capture.

### Bundled CUPTI

Docker builds (`make build-amd64`, the release `parcagpu-<arch>.tar.gz` and
`/usr/lib` in the runtime image) ship two CUPTIs next to
`libparcagpucupti.so`: `libcupti.so.12` (CUDA 12.9) and `libcupti.so.13`
(CUDA 13.0), each with the `libnvperf_host.so` it needs for PC sampling,
renamed `libnvperf_host.so.12` and `.13`. Keep all five files in the same
directory. You don't need to set an environment variable.

parcagpu looks for CUPTI in this order:

1. `libcupti.so` in the directory named by `TRITON_CUPTI_LIB_PATH`. Set it to
   use your own CUPTI instead of the bundled one. If it can't be loaded,
   parcagpu logs a warning and continues.
2. A bundled CUPTI:
   - `libcupti.so.13` when the driver supports CUDA 13 (driver 580 or newer)
     and every GPU is Turing or newer (compute capability 7.5). GPUs released
     after CUDA 12.9 need this one.
   - `libcupti.so.12` otherwise. It needs driver 575.57.08 or newer. If an
     older driver makes CUPTI fail with `CUPTI_ERROR_NOT_INITIALIZED`, point
     `TRITON_CUPTI_LIB_PATH` at a `libcupti.so` that matches the driver.
3. A `libcupti.so` the process has already loaded, then `libcupti.so` on
   `LD_LIBRARY_PATH` and the system library paths. Only builds without the
   bundled files get this far. The bundled CUPTI goes first because an older
   `libcupti.so` on the library path records nothing on newer GPUs.

`PARCAGPU_DEBUG=1` prints the choice. When PC sampling is on, parcagpu also
loads the `libnvperf_host.so` (about 110 MB of memory) in the same directory
as the CUPTI it chose: `libnvperf_host.so.12` or `.13` for the bundled one,
`libnvperf_host.so` otherwise. CUPTI needs it for stall reasons. If that
directory has none, CUPTI looks for `libnvperf_host.so` on `LD_LIBRARY_PATH`
and the system library paths, and without one PC sampling reports no stall
reasons.

An application may load its own `libcupti.so.12` or `.13`, as PyTorch does.
parcagpu still loads its bundled copy, so the process has two. CUPTI takes
one subscriber per process: if the application starts its own CUPTI profiler
(such as `torch.profiler`) first, parcagpu records nothing; if it starts one
later, that profiler fails with `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`.

### Environment variables

Under control from [parca-agent](https://github.com/parca-dev/parca-agent), the
agent's settings replace these (see below). An unrecognized `PARCAGPU_*`
variable or an invalid value is logged as a warning (see [Logging](#logging)).

| Variable | Default | Description |
|---|---|---|
| `PARCAGPU_RATE_LIMIT` | 100 / GPUs | Kernel launches per second per thread that reach the probes (token bucket). Split across the machine's GPUs (`/proc/driver/nvidia/gpus`, not only those the process sees), so with a launching thread per GPU the machine sends about 100/s: 12.5 per thread on 8 GPUs. |
| `PARCAGPU_GRAPH_RATE_LIMIT` | 20 / GPUs | The same for graph launches (`cuGraphLaunch`, `cudaGraphLaunch`), from a separate bucket, so graph replays can't flood the probes and eager bursts can't starve graphs. |
| `PARCAGPU_PC_SAMPLING_RATE` | unset | Set to enable PC sampling: target samples per second, reached by adjusting each window's dice-roll probability. 0 enables it at a fixed probability. Needs access to the GPU performance counters: CAP_SYS_ADMIN, or the driver loaded with `NVreg_RestrictProfilingToAdminUsers=0`; without it parcagpu keeps PC sampling off. |
| `PARCAGPU_SAMPLING_FACTOR` | 20 | PC sampling period, 2^factor cycles per SM, 5-31. Applied instead of CUPTI's per-GPU default (10 on a GB10). |
| `PARCAGPU_PC_HW_BUFFER_MB` | unset | CUPTI's PC sampling hardware buffer in MB, up to 4096. Unset, 0 or out of range: CUPTI's 512 MB. On a GB10, forcing any size has made PC sampling return no data. |
| `PARCAGPU_METADATA_REFRESH_MS` | 30000 | How often cubins, the stall reason map and `gpu_config` are re-emitted. A consumer that attaches while another is attached gets them only then: the USDT semaphore stays at 1. |
| `PARCAGPU_BLOCKING_CALLS` | unset | Any value but empty or `0` enables `cuda_blocking_call`. Off by default because its callbacks run on every covered API call, query polling included, even with no consumer. |
| `PARCAGPU_BLOCKING_THRESHOLD_US` | 10 | Threshold T for `cuda_blocking_call`. Calls blocked at least T are reported with weight = duration, shorter ones with probability duration/T and weight T, so summed weights estimate total blocked time without bias. 0 reports every call. |
| `PARCAGPU_LOG_LEVEL` | `info` | Lowest level on the `log` probe: `debug`, `info`, `warn` or `error`. |
| `PARCAGPU_DEBUG` | unset | Every log record, `trace` included, on stderr. Without it only errors go to stderr. |
| `PARCAGPU_GATE_STATS` | unset | Print the PC sampling gate's counters every 10 s. |

### Runtime control from [parca-agent](https://github.com/parca-dev/parca-agent)

[parca-agent](https://github.com/parca-dev/parca-agent) can configure each
process at runtime over [procctl](https://github.com/parca-dev/procctl)
(vendored as `src/procctl.h`). Each process creates a sealed memfd,
`procctl:parcagpu-v1`, which the agent finds through `/proc`. The agent writes a
configuration and reads back the devices the process uses and the settings in
effect. The schema is `src/parcagpu_ctl.h`: a fixed 32-word config in which
every setting is 0 by default and new ones come with a schema version the agent
checks first. A 0 leaves the setting to the environment.

| Setting | Instead of | Notes |
|---|---|---|
| PC sampling, per device | `PARCAGPU_PC_SAMPLING_RATE` | CUPTI samples one process per GPU at a time, so the agent grants each GPU to one process and rotates it. A granted process samples in windows (30 ms by default) a gap apart, each opening on the first launch after the gap. A context is set up for PC sampling only while its device is granted: on a GB10 the sampler belongs to the first process that enables it, until that process disables it. |
| Sampling factor | `PARCAGPU_SAMPLING_FACTOR` | Applied live. |
| Launch rate limits | `PARCAGPU_RATE_LIMIT`, `PARCAGPU_GRAPH_RATE_LIMIT` | Applied live and reported. |
| PC hardware buffer, 1-4096 MB | `PARCAGPU_PC_HW_BUFFER_MB` | Taken at a context's next PC sampling setup: a granted device's next launch outside a window, otherwise when granted. Reported. |
| Metadata refresh | `PARCAGPU_METADATA_REFRESH_MS` | From the next launch; lowering it refreshes at once. |
| Blocking threshold, 1 µs-1 s | `PARCAGPU_BLOCKING_THRESHOLD_US` | From the next blocking call. The environment's 0 (every call) has no equivalent. |
| Features `kernel_stats` and `blocking_calls` | `PARCAGPU_BLOCKING_CALLS` | Each is off unless the agent sets its bit (`PARCAGPU_FEATURE_*`). Reported. |

parcagpu acknowledges a configuration once it takes effect, so an ack with a
device off means its GPU is free. A small thread does this within ~100 ms even
if the application makes no CUDA calls. It never starts sampling. A device
another process samples, or whose setup failed, is reported unavailable. A
configuration that fails validation is not acknowledged and has no effect. With
no agent, or once the agent's lease expires, parcagpu runs on its environment.

## USDT probes

Defined in `src/probes.d`, provider `parcagpu`. `cbid` is signed: negative for
the driver API, positive for the runtime API. Timestamps are
`CLOCK_MONOTONIC` ns.

| Probe | Arguments | Description |
|---|---|---|
| `cuda_correlation` | correlationId, cbid, name | A launch (kernel or graph), from the launching thread so the CPU stack is the caller's. Rate limited. |
| `kernel_executed` | start, end, correlationId, deviceId, streamId, graphId, graphNodeId, name | Timing of a kernel whose launch passed the rate limit. |
| `activity_batch` | ptrs, count | The same kernels as raw CUPTI activity records. |
| `kernel_stats` | header, entries, count | Exact kernel counts and summed durations per device, stream and kernel name, counted before rate limiting. Deltas, at most once a second and at exit; more than 256 entries split across firings. Under the agent, only with `PARCAGPU_FEATURE_KERNEL_STATS`. Layouts: `src/kernel_stats.h`, `ebpf/cupti_bpf.h`. |
| `cuda_blocking_call` | startNs, endNs, cbid, deviceId, weightNs | A CUDA call that blocked the host thread: GPU waits (synchronize, sync memcpy, free, stream and event queries), device and pinned allocation, VMM map and unmap, module loading, graph instantiate, update and upload. Async enqueues are excluded. Fired inside the call. `deviceId` is `UINT32_MAX` if unknown; `weightNs` is the time the sample stands for (see `PARCAGPU_BLOCKING_THRESHOLD_US`). Opt in: `PARCAGPU_BLOCKING_CALLS`, or the agent's `PARCAGPU_FEATURE_BLOCKING_CALLS`. |
| `pc_sample_batch` | records, count, deviceId | PC sampling records. |
| `stall_reason_map` | names, count | Stall reason names. Metadata. |
| `gpu_config` | deviceId, samplingFactor, clockKHz, smCount | The sampling factor CUPTI applies, with the clock and SM count to turn samples into time. Metadata. |
| `cubin_loaded` | cubinCrc, cubin, cubinSize | A module load. Metadata. |
| `cubin_loaded_file` | cubinCrc, cubin, cubinSize, path | The same, with the module's file when there is one (below). Metadata. |
| `cubin_unloaded` | cubinCrc | A module unload. |
| `device_info` | deviceId, uuid, pciBusId | A device's UUID and PCI bus ID, when the process first uses it, when a consumer attaches, and every 30 s. |
| `nvtx_event_batch` | ptrs, count | NVTX events (`struct nvtx_event`, `src/nvtx.h`), flushed at 256 events or when the oldest is 100 ms old. |
| `nvtx_resource_name` | kind, id, name | NVTX name for an OS thread, CUDA device or stream, or domain (`NVTX_RESOURCE_*`). |
| `log` | level, code, timeNs, component, message | A log record (see [Logging](#logging)). `timeNs` is `CLOCK_REALTIME`. |
| `error` | code, message, component | The `log` probe's warnings and errors, for consumers older than it. |

Metadata probes fire as things load, when a consumer attaches, and every
`PARCAGPU_METADATA_REFRESH_MS`.

### Logging

parcagpu logs to the `log` probe, so that an agent can show what parcagpu does
in each process it attaches to. Levels: 1 debug (per window, context or
call), 2 info (once per process or per change: a startup record with the
version, CUDA and CUPTI versions and what is on, each device, agent
configurations applied, starting and stopping recording every kernel for
`kernel_stats`), 3 warn (something degraded: a feature off, input
ignored, data lost), 4 error (parcagpu isn't working in this process, such as
`cuptiSubscribe` failing because Nsight or DCGM profiling holds CUPTI). Records
at or above `PARCAGPU_LOG_LEVEL` fire while a consumer is attached, at most
100/s for debug and info and 20/s for warnings and errors; a warning reports
what was not sent. `code` is the failed call's result (`CUptiResult`,
`CUresult` or errno), or 0. Messages are at most 255 bytes.

Most failures happen while the driver loads parcagpu, before an agent has
attached to the new process. So parcagpu keeps the last 64 info, warning and
error records and fires them again half a second after a consumer attaches (the
probe's semaphore count goes up). Replayed records keep their `timeNs`, so a
consumer that sees one twice, because another consumer attached, can drop it by
(pid, `timeNs`). Info and higher records logged while a replay is pending go
out with it. The `error` probe gets the same replay of warnings and errors.
`trace` records (per PC sample, per NVTX event, per launch) go only to stderr.

**Cubin files.** Attach to one of `cubin_loaded` and `cubin_loaded_file`,
preferring the second where it exists (older parcagpu lacks it). Only
`cuModuleLoad` gives a path, absolute and resolved in the application's working
directory and mount namespace; `cuModuleLoadData`/`DataEx`, JIT linking and
deferred loads give an empty one. A path is only a hint: PTX, fatbinaries and
shared libraries need not contain the exact image. Validate the whole file
against the loaded image, resolve the path through the process's root, and
expect the file to change or disappear. A missing path must not stop PC offsets,
kernel identity, stall reasons or sample counts from being reported.

## PC sampling and concurrent launches

CUPTI's PC sampling API isn't safe while other threads launch work. On a GB10
with CUPTI 13.0 (no release note up to 13.4 says otherwise):

- `cuptiPCSamplingStart`, `Stop`, `Enable` and `Disable` deadlock against
  another thread in a kernel launch, memset or async memcpy: the PC sampling
  call holds a CUPTI mutex and waits for the driver lock, while the other
  thread holds the driver lock and waits for the mutex. Both collection modes.
  `test/relrace.cu` reproduces it.
- Continuous mode doesn't help: Start/Stop deadlock the same way, draining after
  Disable yields nothing, and leaving it on stalls other threads' launches for
  0.3-3 s at any factor and loses the samples' launch correlation. So parcagpu
  uses short `KERNEL_SERIALIZED` windows, opened on a launch, then stopped and
  drained.
- While a serialized window is open the launching thread holds the driver lock
  and other threads starve: a hang if the owner waits for one of them.

So with PC sampling on, every driver API call except those that only wait,
query or read context state passes through a gate (`src/pc_gate.h`) between its
ENTER and EXIT callbacks. A PC sampling state change runs only with no other
thread inside a gated call, and while a window is open only its owner's calls
are admitted; others wait at ENTER, outside the driver. Every wait is bounded:
a close that takes over 100 ms gives up (and closes fail at once for 2 s), a
call waiting for a close goes in after 250 ms, and a call waiting past a
window's deadline (length + 20 ms) closes it or stops honoring it. The gate
fails open, trading a possible stall for a hang that can't be undone.

A driver call from parcagpu during stream capture can invalidate it, so there
are no PC sampling state changes while any thread captures:
`cuStreamBeginCapture` waits up to 1 s for changes in flight and stops any open
window, and agent configurations that must wait are retried.

The cost: each window stalls other threads for 50-300 ms, gate or no gate. That
is why PC sampling is opt in, and under agent control runs only when granted.

## Building

```bash
make local         # build-local/lib/libparcagpucupti.so (CMake, RelWithDebInfo)
make debug         # Debug build, same place
make build-amd64   # Docker build of the .so (also build-arm64, build-all)
make cross         # Runtime image for both architectures (buildx cache)
make docker-push   # Push the multi-arch image to ghcr.io (IMAGE, IMAGE_TAG)
make format        # clang-format
make clean
```

## Testing

```bash
make test                 # Mock CUPTI harness
make test-nvtx            # NVTX collector
make test-machine-gpus    # Machine GPU count and the rate limit split
make test-profiling-access # Driver profiling restriction check
make test-gate            # PC sampling gate, under ThreadSanitizer
make test-pc-contexts     # PC sampling start/stop across mock contexts
make test-agent-ctl       # Control over procctl, from a Go agent
make test-graph-rate-limit # Graph rate limit at a production launch rate
make docker-test-run      # make test in a container
make test-pc-mock         # Mock PC sampling + BPF activity parser (root)
make test-blocking-mock   # cuda_blocking_call with bpftrace (root)
make test-log             # Log replay to a late consumer, failed start (root)
make test-nvtx-real       # Real NVTX clients + BPF parser (root; NVTX_PYTHON=... adds Python)
make test-multi           # Mock harness + BPF parser in parallel (root)
make test-pc-real         # Real PC sampling on microbenchmarks/ (root, GPU)
```

Only `test-pc-real` needs a GPU. To watch a running process:

```bash
sudo bpftrace -p <PID> parcagpu.bt   # -p, or probes behind semaphores stay off
make bpf-test && sudo test/bpf/activity_parser -pid <PID> -lib <path/to/libparcagpucupti.so> -v
```

The activity parser attaches to every probe and resolves PC samples to source
lines with `llvm-dwarfdump`.

## Requirements

- CUDA Toolkit (CUPTI), CMake, dtrace (systemtap-sdt-dev)
- For the BPF tests and tools: clang, libbpf-dev, bpftool, Go 1.25+, bpftrace

## Layout

```
src/
  cupti.cpp           CUPTI callbacks, activity and probes
  pc_sampling.cpp     PC sampling
  pc_gate.cpp         PC sampling gate
  agent_ctl.cpp       Control from parca-agent; schema in parcagpu_ctl.h
  procctl.h           Vendored procctl (process side)
  kernel_stats.cpp    kernel_stats probe
  device_info.cpp     device_info probe
  nvtx_*.cpp          NVTX injection and collector
  env_config.cpp      Environment variable checks
  log.cpp             Log channel: log probe, replay, stderr
  probes.d            USDT probe definitions
ebpf/cupti_bpf.h      Structs shared with BPF consumers
test/                 Mock CUPTI and CUDA, test scripts, agentctl/ (Go), bpf/ (activity parser)
microbenchmarks/      CUDA programs for test-pc-real
parcagpu.bt           bpftrace monitor
```
