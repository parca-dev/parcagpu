// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// EXPERIMENT: time sampled kernel launches with a pair of CUDA events instead
// of CUPTI CONCURRENT_KERNEL activity, so the workload pays per sampled launch
// rather than per kernel.
//
//   PARCAGPU_EVENT_TIMING=1  events replace activity (activity never enabled)
//   PARCAGPU_EVENT_TIMING=2  both: activity stays on and drives the probes;
//                            every event-timed launch and every CUPTI record
//                            for it is logged to stderr ("PGEV ...") so the
//                            two durations can be compared offline
//   PARCAGPU_EVENT_SAMPLE_P  probability a launch is considered (default 0.05);
//                            the launch token buckets still cap the rate
#ifndef PARCAGPU_EVENT_TIMING_H_
#define PARCAGPU_EVENT_TIMING_H_

#include <cstdint>
#include <cupti.h>

namespace parcagpu::event_timing {

void init();
int mode();          // 0 off, 1 replace activity, 2 compare
double sampleP();

// Driver launch ENTER: record the start event on the launch's stream.
// Returns false (nothing recorded) for launch kinds it can't time.
bool begin(const CUpti_CallbackData *cb, CUpti_CallbackId cbid,
           uint32_t deviceId);
// Driver launch EXIT for the launch begin() accepted on this thread.
void end(const CUpti_CallbackData *cb);
// Correlation id begin() accepted on this thread and not yet consumed.
uint32_t pendingCorrelation();
void consumeCorrelation();

// Poll finished event pairs; emit kernel_executed (mode 1) or log (mode 2).
void reap();

// Mode 2: log a CUPTI kernel record if its launch was event-timed.
void logCuptiRecord(uint32_t correlationId, uint64_t start, uint64_t end,
                    uint32_t graphId);

} // namespace parcagpu::event_timing

#endif
