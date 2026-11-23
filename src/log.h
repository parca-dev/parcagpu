// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// parcagpu's log channel. One call writes a record to up to three places:
//
//  - the `log` USDT probe, for parca-agent (and bpftrace): records at or
//    above PARCAGPU_LOG_LEVEL (default info), while a consumer is attached.
//    Rate limited, so a log loop can't crowd GPU events out of the agent's
//    ring buffer. Warnings and errors also fire the older `error` probe,
//    which agents before the `log` probe consume.
//  - a ring of the last kReplayRecords info-and-above records, fired again on
//    both probes half a second after a consumer attaches (the probe's
//    semaphore count goes up). Most failures happen while the CUDA driver loads
//    parcagpu, usually before an agent has attached to the new process; without
//    the replay it would never see them. Replayed records keep their time, so a
//    consumer that sees one twice can drop it by (pid, time_ns).
//  - stderr: everything with PARCAGPU_DEBUG set, else errors only.
//
// The first record a process writes is preceded by a build record: info,
// component "build", message buildInfo() (version.h), e.g.
//   version=v0.4.0-4-gabc1234 sha=<40 hex> dirty=0 build_type=RelWithDebInfo
//   cuda_headers=12.8 cupti_api=24
// It goes to the log probe whatever PARCAGPU_LOG_LEVEL says, never to the
// error probe, and is kept apart from the ring: every replay starts with it,
// however many records were written since.
//
// Levels (the probe's `level` argument):
//   trace  stderr only, never on the probe: per-event detail (per PC, per
//          NVTX event) that would flood any channel but a terminal.
//   debug  per window / per context / per call detail.
//   info   once-per-process and configuration changes; what an agent log
//          should show for a healthy process.
//   warn   something degraded (a feature off, input ignored), the process
//          is otherwise fine.
//   error  parcagpu isn't doing its job in this process.
//
// `component` is a short name of the part of parcagpu that logs
// ("cupti", "pc_sampling", "agent_ctl", ...). `code` is the result of the
// failed call (a CUptiResult, CUresult or errno) where there is one, else 0.
#ifndef PARCAGPU_LOG_H_
#define PARCAGPU_LOG_H_

#include <cstddef>
#include <cstdint>

namespace parcagpu {
namespace log {

enum Level : uint32_t {
  kTrace = 0,
  kDebug = 1,
  kInfo = 2,
  kWarn = 3,
  kError = 4,
};

// Message bytes carried by the probe, terminator included; longer messages
// are cut. Matches the agent's error event buffer.
constexpr size_t kMessageMax = 256;
// Records kept for replay.
constexpr size_t kReplayRecords = 64;
// Probe rate limit, records per second (a second's worth of burst).
constexpr double kProbeRate = 100.0;

// Whether a record at this level would go anywhere. Cheap: the PLOG_* macros
// check it before formatting.
bool enabled(Level level);

// Format and write one record. Use the PLOG_* macros.
void write(Level level, int32_t code, const char *component, const char *fmt,
           ...) __attribute__((format(printf, 4, 5)));

// Replay the ring if a consumer attached since the last check. Called on
// CUDA API callbacks and from the agent control thread; write() checks too.
void poll();

// Start a thread that calls poll() every 100 ms, for a process in which
// parcagpu failed to start: no CUDA callbacks or agent control thread poll
// there, and its failure is what a late consumer most needs replayed. Once.
void startPoller();

// Whether s names a level (PARCAGPU_LOG_LEVEL's values).
bool parseLevel(const char *s);

// Whether PARCAGPU_DEBUG is set (stderr gets every level).
bool stderrDebug();

const char *levelName(Level level);

// Test hook: the probe threshold, as from PARCAGPU_LOG_LEVEL.
Level probeLevel();

} // namespace log
} // namespace parcagpu

#define PLOG_(lvl, code, comp, ...)                                            \
  do {                                                                         \
    if (::parcagpu::log::enabled(lvl))                                         \
      ::parcagpu::log::write(lvl, code, comp, __VA_ARGS__);                    \
  } while (0)

#define PLOG_TRACE(comp, ...)                                                  \
  PLOG_(::parcagpu::log::kTrace, 0, comp, __VA_ARGS__)
#define PLOG_DEBUG(comp, ...)                                                  \
  PLOG_(::parcagpu::log::kDebug, 0, comp, __VA_ARGS__)
#define PLOG_INFO(comp, ...) PLOG_(::parcagpu::log::kInfo, 0, comp, __VA_ARGS__)
#define PLOG_WARN(code, comp, ...)                                             \
  PLOG_(::parcagpu::log::kWarn, code, comp, __VA_ARGS__)
#define PLOG_ERROR(code, comp, ...)                                            \
  PLOG_(::parcagpu::log::kError, code, comp, __VA_ARGS__)

#endif // PARCAGPU_LOG_H_
