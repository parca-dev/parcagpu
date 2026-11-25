// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include "log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <signal.h>
#include <strings.h>
#include <time.h>

// USDT probes — before any header that might include <sys/sdt.h>.
#include "probes.h"

#include "pc_sampling.h" // fireError
#include "version.h"

namespace parcagpu {
namespace log {
namespace {

constexpr size_t kComponentMax = 32;

struct Config {
  Level probe;    // threshold for the log probe
  bool stderrAll; // PARCAGPU_DEBUG: every level to stderr
};

bool parse(const char *s, Level *out) {
  static const struct {
    const char *name;
    Level level;
  } names[] = {
      // trace never reaches the probe; as a threshold it means debug.
      {"trace", kDebug}, {"debug", kDebug},  {"info", kInfo},
      {"warn", kWarn},   {"warning", kWarn}, {"error", kError},
  };
  for (const auto &n : names) {
    if (strcasecmp(s, n.name) == 0) {
      *out = n.level;
      return true;
    }
  }
  return false;
}

pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

// A child of fork() has only the forking thread; a mutex another thread held
// at the fork would stay locked forever.
void atforkChild() { g_mutex = PTHREAD_MUTEX_INITIALIZER; }

const Config &config() {
  static const Config c = [] {
    Config c{kInfo, getenv("PARCAGPU_DEBUG") != nullptr};
    if (const char *v = getenv("PARCAGPU_LOG_LEVEL"))
      parse(v, &c.probe); // an invalid value is reported by env_config
    pthread_atfork(nullptr, nullptr, atforkChild);
    return c;
  }();
  return c;
}

// The USDT semaphores count attached uprobe consumers; the kernel updates
// them from outside this address space.
uint16_t sem(unsigned short &s) {
  return __atomic_load_n(reinterpret_cast<uint16_t *>(&s), __ATOMIC_RELAXED);
}
uint16_t logSem() { return sem(parcagpu_log_semaphore); }
uint16_t errorSem() { return sem(parcagpu_error_semaphore); }

uint64_t realtimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}

uint64_t monotonicNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}

// Out of line, so the probe has exactly one site.
__attribute__((noinline)) void fireLog(uint32_t level, int32_t code,
                                       uint64_t timeNs, const char *component,
                                       const char *message) {
  PARCAGPU_LOG(level, code, timeNs, component, message);
}

struct Record {
  Level level;
  int32_t code;
  uint64_t timeNs;
  char component[kComponentMax];
  char message[kMessageMax];
};

// The build record: written before any other record, exempt from the probe
// threshold, and kept outside the ring, so every replay starts with it however
// much was logged since. Guarded by g_mutex.
Record g_build;
bool g_buildWritten = false;

// Guarded by g_mutex.
Record g_ring[kReplayRecords];
size_t g_ringNext = 0; // total records pushed; next slot is % kReplayRecords

// A token bucket starting full (TokenBucket starts with one token, and a
// process logs most at startup). Guarded by g_mutex.
struct Bucket {
  double rate;
  double tokens;
  uint64_t lastNs;
  uint64_t suppressed;   // records not sent since the last notice
  uint64_t lastNoticeNs; // monotonic

  bool take() {
    const uint64_t now = monotonicNs();
    if (lastNs != 0) {
      tokens += (now - lastNs) / 1e9 * rate;
      if (tokens > rate)
        tokens = rate;
    }
    lastNs = now;
    if (tokens < 1.0) {
      suppressed++;
      return false;
    }
    tokens -= 1.0;
    return true;
  }
};
// Warnings and errors get their own bucket so that a flood of debug records
// can't push them out.
Bucket g_lowBucket{kProbeRate, kProbeRate, 0, 0, 0};
Bucket g_highBucket{kProbeRate / 5, kProbeRate / 5, 0, 0, 0};

std::atomic<uint16_t> g_prevLogSem{0};
std::atomic<uint16_t> g_prevErrorSem{0};

// A replay waits this long after the join that asks for it. The kernel bumps
// a probe's semaphore when a consumer creates the uprobe, and the consumer
// attaches its program to it after: a replay fired at once can land in
// between and be lost (bpftrace loses it most of the time).
constexpr uint64_t kReplayDelayNs = 500'000'000ULL;

// A replay waiting to be fired. Guarded by g_mutex, except g_replayPending,
// which poll() reads first without it.
std::atomic<bool> g_replayPending{false};
bool g_replayLog = false;   // replay on the log probe
bool g_replayError = false; // replay warnings and errors on the error probe
std::atomic<uint64_t> g_replayDueNs{0}; // monotonic

void fire(const Record &r, bool toLog, bool toError) {
  if (toLog)
    fireLog(r.level, r.code, r.timeNs, r.component, r.message);
  if (toError)
    fireError(r.code, r.message, r.component);
}

// Schedule a replay of the ring on each probe whose consumer count went up,
// and fire one that is due.
void replayLocked() {
  const uint16_t ls = logSem();
  const uint16_t es = errorSem();
  const bool logJoin =
      ls > g_prevLogSem.exchange(ls, std::memory_order_relaxed);
  const bool errorJoin =
      es > g_prevErrorSem.exchange(es, std::memory_order_relaxed);
  const uint64_t now = monotonicNs();
  if (logJoin || errorJoin) {
    g_replayLog |= logJoin;
    g_replayError |= errorJoin;
    g_replayDueNs.store(now + kReplayDelayNs, std::memory_order_relaxed);
    g_replayPending.store(true, std::memory_order_relaxed);
    return;
  }
  if (!g_replayPending.load(std::memory_order_relaxed) ||
      now < g_replayDueNs.load(std::memory_order_relaxed))
    return;
  const Level threshold = config().probe;
  // The build record first, whatever the threshold.
  if (g_buildWritten)
    fire(g_build, g_replayLog && logSem() > 0, false);
  // The whole ring, records kept since the join included: write() holds
  // those back from the probes being replayed to, so each goes out once.
  const size_t begin =
      g_ringNext > kReplayRecords ? g_ringNext - kReplayRecords : 0;
  for (size_t i = begin; i < g_ringNext; i++) {
    const Record &r = g_ring[i % kReplayRecords];
    fire(r, g_replayLog && r.level >= threshold && logSem() > 0,
         g_replayError && r.level >= kWarn && errorSem() > 0);
  }
  g_replayLog = g_replayError = false;
  g_replayPending.store(false, std::memory_order_relaxed);
}

void copyStr(char *dst, size_t cap, const char *src) {
  if (!src)
    src = "";
  size_t n = strnlen(src, cap - 1);
  memcpy(dst, src, n);
  dst[n] = '\0';
}

void printStderr(const Record &r) {
  fprintf(stderr, "[%lu.%09lu] [PARCAGPU] %s %s: %s\n",
          (unsigned long)(r.timeNs / 1000000000ULL),
          (unsigned long)(r.timeNs % 1000000000ULL), levelName(r.level),
          r.component, r.message);
}

// Keep r for replay and fire it where it goes now. pinned: the build record,
// kept apart from the ring and sent whatever the probe threshold.
void submitLocked(const Record &r, bool pinned) {
  const Level level = r.level;
  const Config &c = config();
  if (level >= kInfo && !pinned)
    g_ring[g_ringNext++ % kReplayRecords] = r;
  // A record kept for replay waits for a pending replay to a probe rather
  // than fire now, maybe before the new consumer's program is attached.
  const bool held =
      level >= kInfo && g_replayPending.load(std::memory_order_relaxed);
  const bool toLog =
      (pinned || level >= c.probe) && logSem() > 0 && !(held && g_replayLog);
  const bool toError =
      level >= kWarn && errorSem() > 0 && !(held && g_replayError);
  if (toLog || toError) {
    Bucket &b = level >= kWarn ? g_highBucket : g_lowBucket;
    if (b.take()) {
      // At most one notice a second about what wasn't sent.
      if (b.suppressed && b.lastNs - b.lastNoticeNs >= 1000000000ULL) {
        Record s{kWarn, 0, r.timeNs, "log", ""};
        snprintf(s.message, sizeof(s.message),
                 "%lu %s records not sent (over %.0f/s)",
                 (unsigned long)b.suppressed,
                 &b == &g_highBucket ? "warn/error" : "debug/info", b.rate);
        b.suppressed = 0;
        b.lastNoticeNs = b.lastNs;
        fire(s, logSem() > 0, false);
      }
      fire(r, toLog, toError);
    }
  }
}

} // namespace

bool enabled(Level level) {
  if (level >= kInfo)
    return true; // kept for replay
  const Config &c = config();
  if (c.stderrAll)
    return true;
  return level != kTrace && level >= c.probe && logSem() > 0;
}

void write(Level level, int32_t code, const char *component, const char *fmt,
           ...) {
  Record r;
  r.level = level;
  r.code = code;
  r.timeNs = realtimeNs();
  copyStr(r.component, sizeof(r.component), component);
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(r.message, sizeof(r.message), fmt, ap);
  va_end(ap);
  // Records are single lines.
  for (size_t n = strlen(r.message); n > 0 && r.message[n - 1] == '\n'; n--)
    r.message[n - 1] = '\0';

  const Config &c = config();
  if (c.stderrAll || level >= kError)
    printStderr(r);
  if (level == kTrace)
    return;

  pthread_mutex_lock(&g_mutex);
  replayLocked();
  if (!g_buildWritten) {
    g_buildWritten = true;
    g_build = Record{kInfo, 0, r.timeNs, "build", ""};
    copyStr(g_build.message, sizeof(g_build.message), buildInfo());
    if (c.stderrAll)
      printStderr(g_build);
    submitLocked(g_build, true);
  }
  submitLocked(r, false);
  pthread_mutex_unlock(&g_mutex);
}

void poll() {
  uint16_t prevLog = g_prevLogSem.load(std::memory_order_relaxed);
  uint16_t prevError = g_prevErrorSem.load(std::memory_order_relaxed);
  const uint16_t ls = logSem();
  const uint16_t es = errorSem();
  if (ls <= prevLog && es <= prevError) {
    if (g_replayPending.load(std::memory_order_relaxed)) {
      if (monotonicNs() < g_replayDueNs.load(std::memory_order_relaxed))
        return; // not due yet
    } else {
      // No join. Record a detach, so that the next attach counts as one; a
      // join racing this still shows as a count above what's stored.
      if (ls < prevLog)
        g_prevLogSem.compare_exchange_strong(prevLog, ls);
      if (es < prevError)
        g_prevErrorSem.compare_exchange_strong(prevError, es);
      return;
    }
  }
  pthread_mutex_lock(&g_mutex);
  replayLocked();
  pthread_mutex_unlock(&g_mutex);
}

void startPoller() {
  static std::atomic<bool> started{false};
  if (started.exchange(true))
    return;
  // Block signals in the thread: they belong to the application's threads.
  sigset_t all, old;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &old);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  pthread_t t;
  pthread_create(
      &t, &attr,
      [](void *) -> void * {
        for (;;) {
          struct timespec ts = {0, 100'000'000L};
          nanosleep(&ts, nullptr);
          poll();
        }
        return nullptr;
      },
      nullptr);
  pthread_attr_destroy(&attr);
  pthread_sigmask(SIG_SETMASK, &old, nullptr);
}

bool stderrDebug() { return config().stderrAll; }

Level probeLevel() { return config().probe; }

bool parseLevel(const char *s) {
  Level l;
  return parse(s, &l);
}

const char *levelName(Level level) {
  switch (level) {
  case kTrace:
    return "TRACE";
  case kDebug:
    return "DEBUG";
  case kInfo:
    return "INFO";
  case kWarn:
    return "WARN";
  case kError:
    return "ERROR";
  }
  return "?";
}

} // namespace log
} // namespace parcagpu
