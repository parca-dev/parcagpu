// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include "probes.h"

#include "kernel_stats.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "agent_ctl.h"
#include "parcagpu_ctl.h"
#include "pc_sampling.h" // DEBUG_PRINTF

// Single call site for the probe (see the fireX wrappers in pc_sampling.cpp).
__attribute__((noinline)) void
fireKernelStats(const parcagpu::kernel_stats::Header *header,
                const parcagpu::kernel_stats::Entry **entries, uint32_t count) {
  PARCAGPU_KERNEL_STATS(header, reinterpret_cast<const void **>(entries),
                        count);
}

namespace parcagpu::kernel_stats {
namespace {

constexpr uint64_t kEmitPeriodNs = 1000ULL * 1000 * 1000;

// CUPTI shares one name string across all records of the same kernel, so
// the pointer identifies the kernel within an interval. The name is copied
// on first sight in each interval, so a module unloaded mid-interval cannot
// leave a dangling pointer for the probe to read.
struct Key {
  uint32_t deviceId;
  uint32_t streamId;
  const char *name;
  bool operator==(const Key &o) const {
    return deviceId == o.deviceId && streamId == o.streamId && name == o.name;
  }
};

struct KeyHash {
  size_t operator()(const Key &k) const {
    size_t h = std::hash<const void *>()(k.name);
    h ^= (uint64_t(k.deviceId) << 32 | k.streamId) + 0x9e3779b97f4a7c15ULL +
         (h << 6) + (h >> 2);
    return h;
  }
};

struct Slot {
  std::string name;
  uint64_t executions = 0;
  uint64_t totalNs = 0;
};

struct Interval {
  std::unordered_map<Key, Slot, KeyHash> slots;
  uint64_t startNs = UINT64_MAX;
  uint64_t endNs = 0;
};

std::mutex g_mutex;
Interval g_interval;       // guarded by g_mutex
uint64_t g_lastEmitNs = 0; // guarded by g_mutex
uint32_t g_seq = 0;        // guarded by g_mutex

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}

void fire(Interval &iv, uint32_t seq) {
  Header header{};
  header.startNs = iv.startNs;
  header.endNs = iv.endNs;
  header.seq = seq;

  std::vector<Entry> entries;
  entries.reserve(iv.slots.size());
  for (const auto &[key, slot] : iv.slots) {
    entries.push_back(Entry{key.deviceId, key.streamId, slot.executions,
                            slot.totalNs, slot.name.c_str()});
  }
  std::vector<const Entry *> ptrs;
  ptrs.reserve(entries.size());
  for (const auto &e : entries)
    ptrs.push_back(&e);

  for (size_t off = 0; off < ptrs.size(); off += kMaxEntriesPerFiring) {
    const uint32_t n =
        uint32_t(std::min<size_t>(kMaxEntriesPerFiring, ptrs.size() - off));
    fireKernelStats(&header, ptrs.data() + off, n);
  }
  DEBUG_PRINTF("[PARCAGPU] kernel_stats: seq=%u entries=%zu\n", seq,
               entries.size());
}

} // namespace

bool enabled() {
  return agentCtlFeature(PARCAGPU_FEATURE_KERNEL_STATS, /*envDefault=*/true);
}

void record(const CUpti_ActivityKernel5 *k) {
  if (!PARCAGPU_KERNEL_STATS_ENABLED() || !enabled())
    return;
  const uint64_t dur = k->end > k->start ? k->end - k->start : 0;
  const Key key{k->deviceId, k->streamId, k->name};

  std::lock_guard<std::mutex> lock(g_mutex);
  auto [it, inserted] = g_interval.slots.try_emplace(key);
  if (inserted)
    it->second.name = k->name ? k->name : "";
  it->second.executions++;
  it->second.totalNs += dur;
  g_interval.startNs = std::min(g_interval.startNs, k->start);
  g_interval.endNs = std::max(g_interval.endNs, k->end);
}

void maybeEmit(bool force) {
  Interval iv;
  uint32_t seq;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const uint64_t now = nowNs();
    if (!force && now - g_lastEmitNs < kEmitPeriodNs)
      return;
    if (g_interval.slots.empty())
      return;
    g_lastEmitNs = now;
    seq = g_seq++;
    std::swap(iv, g_interval);
  }
  // Fire outside the lock; iv owns the name strings the probe points at.
  // A consumer that detached since record() simply misses this firing.
  fire(iv, seq);
}

} // namespace parcagpu::kernel_stats
