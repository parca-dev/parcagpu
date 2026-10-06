// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_RUNTIME_CONFIG_H_
#define PARCAGPU_RUNTIME_CONFIG_H_

#include <atomic>
#include <cstdint>
#include <time.h>

namespace parcagpu {

// Settings that can change while the process runs. They start from the
// environment; PARCAGPU_CONFIG_FILE, if set, names a file of KEY=VALUE lines
// (the same names as the env vars, PARCAGPU_ prefix optional, # comments)
// that is re-read about once a second. A key in the file overrides the env
// var; removing it (or the file) reverts to the env value.
//
// PARCAGPU_SAMPLING_FACTOR and PARCAGPU_PC_HW_BUFFER_MB are env-only, and
// PC sampling must be enabled at startup (PARCAGPU_PC_SAMPLING_RATE in the
// env) for PC_SAMPLING_RATE in the file to have any effect.
struct LiveSettings {
  bool debug;
  double rateLimit;      // PARCAGPU_RATE_LIMIT
  double graphRateLimit; // PARCAGPU_GRAPH_RATE_LIMIT
  double pcSamplingRate; // PARCAGPU_PC_SAMPLING_RATE; 0 pauses PC sampling
  bool enabled;          // PARCAGPU_ENABLED
  double dutyCycle;      // PARCAGPU_DUTY_CYCLE, fraction of each period on
  double dutyPeriodSec;  // PARCAGPU_DUTY_PERIOD, seconds
};

// Whether profiling is on right now. When off, the shim behaves as if no
// profiler were attached: no activity buffers, no correlation probes and no
// new PC sampling windows. It is off while PARCAGPU_ENABLED=0, and outside
// the first PARCAGPU_DUTY_CYCLE fraction of each PARCAGPU_DUTY_PERIOD.
// Duty windows are aligned to wall-clock time, so every process on a node
// (e.g. the ranks of a tensor-parallel job) is on at the same time.
extern std::atomic<bool> g_profilingActive;
inline bool profilingActive() {
  return g_profilingActive.load(std::memory_order_relaxed);
}

// Read the env settings and settle the gate and kernel activity recording.
// Call once, from CuptiProfiler::initialize() after kernel activity is
// enabled. There is no background thread: runtimeConfigTick(), called on
// every CUPTI callback, re-reads the config file, follows the duty cycle and
// turns kernel activity recording on and off as profilers attach and detach.
void startRuntimeConfig();

// When the next config step is due (CLOCK_REALTIME ns); UINT64_MAX while a
// step runs or before startRuntimeConfig().
extern std::atomic<uint64_t> g_nextConfigStepNs;
void runtimeConfigStep();

// Cheap enough for every callback: one coarse clock read and a load.
inline void runtimeConfigTick() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME_COARSE, &ts);
  uint64_t now = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
  if (now >= g_nextConfigStepNs.load(std::memory_order_relaxed))
    runtimeConfigStep();
}

// Defined in cupti.cpp, which owns the state these settings feed.
LiveSettings currentLiveSettings(); // the rate/debug fields as set from env
void applyLiveSettings(const LiveSettings &s);
// Called on each off->on transition of the gate.
void onProfilingResumed();
// Whether a profiler is consuming the kernel probes.
bool profilerAttached();
// Enable or disable CUPTI kernel activity recording. Returns false if CUPTI
// refused or the profiler is shutting down.
bool setKernelActivity(bool on);

} // namespace parcagpu

#endif // PARCAGPU_RUNTIME_CONFIG_H_
