// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include "runtime_config.h"
#include "pc_sampling.h" // DEBUG_PRINTF, fireError

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <time.h>

namespace parcagpu {

std::atomic<bool> g_profilingActive{true};

static constexpr uint64_t kNsPerSec = 1'000'000'000ULL;
// How often the config file is checked for changes.
static constexpr uint64_t kFileCheckNs = kNsPerSec;
// How often to check whether a profiler has attached or detached.
static constexpr uint64_t kAttachPollNs = kNsPerSec / 4;
static constexpr double kMinDutyPeriodSec = 0.1;
static constexpr size_t kMaxConfigFileBytes = 64 * 1024;

static uint64_t realtimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (uint64_t)ts.tv_sec * kNsPerSec + ts.tv_nsec;
}

static bool parseBool(const char *v, bool &out) {
  if (!strcasecmp(v, "1") || !strcasecmp(v, "true") || !strcasecmp(v, "yes") ||
      !strcasecmp(v, "on")) {
    out = true;
    return true;
  }
  if (!strcasecmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "no") ||
      !strcasecmp(v, "off")) {
    out = false;
    return true;
  }
  return false;
}

static bool parseDouble(const char *v, double &out) {
  char *end = nullptr;
  errno = 0;
  double d = std::strtod(v, &end);
  if (end == v || *end != '\0' || errno != 0 || !std::isfinite(d))
    return false;
  out = d;
  return true;
}

static void reportInvalid(const char *source, const char *key,
                          const char *value, const char *why) {
  char msg[256];
  snprintf(msg, sizeof(msg), "%s=%s %s", key, value, why);
  DEBUG_PRINTF("[PARCAGPU] Warning: %s: %s, ignoring\n", source, msg);
  fireError(0, msg, source);
}

// Apply one KEY=VALUE (key without the PARCAGPU_ prefix) to s. Returns false
// if the key isn't a live setting. fromFile selects the file's semantics for
// DEBUG (a boolean) over the env's (set means on).
static bool applyKey(LiveSettings &s, const char *source, bool fromFile,
                     const char *key, const char *value) {
  double d;
  bool b;
  if (!strcmp(key, "DEBUG")) {
    if (!fromFile)
      s.debug = true;
    else if (parseBool(value, b))
      s.debug = b;
    else
      reportInvalid(source, key, value, "is not a boolean");
  } else if (!strcmp(key, "RATE_LIMIT") || !strcmp(key, "GRAPH_RATE_LIMIT")) {
    if (parseDouble(value, d) && d > 0)
      (key[0] == 'R' ? s.rateLimit : s.graphRateLimit) = d;
    else
      reportInvalid(source, key, value, "must be > 0");
  } else if (!strcmp(key, "PC_SAMPLING_RATE")) {
    if (parseDouble(value, d) && d >= 0)
      s.pcSamplingRate = d;
    else
      reportInvalid(source, key, value, "must be >= 0");
  } else if (!strcmp(key, "ENABLED")) {
    if (parseBool(value, b))
      s.enabled = b;
    else
      reportInvalid(source, key, value, "is not a boolean");
  } else if (!strcmp(key, "DUTY_CYCLE")) {
    if (parseDouble(value, d) && d > 0 && d <= 1)
      s.dutyCycle = d;
    else
      reportInvalid(source, key, value, "must be in (0, 1]");
  } else if (!strcmp(key, "DUTY_PERIOD")) {
    if (parseDouble(value, d) && d >= kMinDutyPeriodSec)
      s.dutyPeriodSec = d;
    else
      reportInvalid(source, key, value, "must be >= 0.1 (seconds)");
  } else {
    return false;
  }
  return true;
}

// Settings with no config file: the env values. The rate/debug fields were
// already parsed (and validated) by init_debug; the gate fields are new here.
static LiveSettings envSettings() {
  LiveSettings s = currentLiveSettings();
  s.enabled = true;
  s.dutyCycle = 1.0;
  s.dutyPeriodSec = 10.0;
  for (const char *key : {"ENABLED", "DUTY_CYCLE", "DUTY_PERIOD"}) {
    std::string name = std::string("PARCAGPU_") + key;
    if (const char *v = getenv(name.c_str()))
      applyKey(s, "env_config", false, key, v);
  }
  return s;
}

static char *trim(char *s) {
  while (*s == ' ' || *s == '\t' || *s == '\r')
    ++s;
  char *e = s + strlen(s);
  while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r'))
    *--e = '\0';
  return s;
}

// Overlay the file's KEY=VALUE lines onto base.
static LiveSettings parseConfig(char *text, const LiveSettings &base) {
  static constexpr const char *kSource = "config_file";
  LiveSettings s = base;
  for (char *save = nullptr, *line = strtok_r(text, "\n", &save); line;
       line = strtok_r(nullptr, "\n", &save)) {
    line = trim(line);
    if (*line == '\0' || *line == '#')
      continue;
    char *eq = strchr(line, '=');
    if (!eq) {
      reportInvalid(kSource, line, "", "is not KEY=VALUE");
      continue;
    }
    *eq = '\0';
    char *key = trim(line);
    char *value = trim(eq + 1);
    if (!strncmp(key, "PARCAGPU_", 9))
      key += 9;
    if (applyKey(s, kSource, true, key, value))
      continue;
    if (!strcmp(key, "SAMPLING_FACTOR") || !strcmp(key, "PC_HW_BUFFER_MB") ||
        !strcmp(key, "CONFIG_FILE"))
      reportInvalid(kSource, key, value, "can only be set in the environment");
    else
      reportInvalid(kSource, key, value, "is not a known setting");
  }
  return s;
}

namespace {

struct FileStamp {
  bool present = false;
  dev_t dev = 0;
  ino_t ino = 0;
  off_t size = 0;
  struct timespec mtime = {};

  bool operator==(const FileStamp &o) const {
    return present == o.present && dev == o.dev && ino == o.ino &&
           size == o.size && mtime.tv_sec == o.mtime.tv_sec &&
           mtime.tv_nsec == o.mtime.tv_nsec;
  }
};

struct ConfigThread {
  const char *path = nullptr; // PARCAGPU_CONFIG_FILE, or null
  LiveSettings env{};         // what keys absent from the file revert to
  LiveSettings cur{};
  FileStamp stamp{};
};

} // namespace

static void logSettings(const char *why, const LiveSettings &s) {
  DEBUG_PRINTF("[PARCAGPU] settings (%s): enabled=%d duty_cycle=%.3f "
               "duty_period=%.1fs rate_limit=%.1f graph_rate_limit=%.1f "
               "pc_sampling_rate=%.1f debug=%d\n",
               why, s.enabled, s.dutyCycle, s.dutyPeriodSec, s.rateLimit,
               s.graphRateLimit, s.pcSamplingRate, s.debug);
}

// Re-read the config file if it changed (a ConfigMap update swaps a symlink,
// which changes the inode stat() follows). Returns true if settings changed.
static bool reloadIfChanged(ConfigThread &t) {
  FileStamp st;
  struct stat sb;
  if (stat(t.path, &sb) == 0) {
    st.present = true;
    st.dev = sb.st_dev;
    st.ino = sb.st_ino;
    st.size = sb.st_size;
    st.mtime = sb.st_mtim;
  }
  if (st == t.stamp)
    return false;
  t.stamp = st;

  if (!st.present) {
    t.cur = t.env;
    logSettings("config file absent", t.cur);
    return true;
  }
  FILE *f = fopen(t.path, "r");
  if (!f) {
    // Keep the current settings; retry once the file changes again.
    DEBUG_PRINTF("[PARCAGPU] can't open config file %s: %s\n", t.path,
                 strerror(errno));
    return false;
  }
  static char buf[kMaxConfigFileBytes + 1];
  size_t n = fread(buf, 1, kMaxConfigFileBytes, f);
  fclose(f);
  buf[n] = '\0';
  t.cur = parseConfig(buf, t.env);
  logSettings("config file", t.cur);
  return true;
}

static void setActive(bool on) {
  if (g_profilingActive.exchange(on, std::memory_order_relaxed) != on)
    DEBUG_PRINTF("[PARCAGPU] profiling %s\n", on ? "on" : "off");
}

// Record kernel activity only while profiling is on and a profiler is
// attached to consume it. Refusing activity buffers instead is not enough:
// CUPTI keeps recording, delivers the backlog once buffers return, and
// launches stay slower from then on -- e.g. after a profiler attaches to an
// already running process.
static void updateKernelActivity() {
  static bool recording = true; // initialize() enabled it
  bool want = profilingActive() && profilerAttached();
  if (want != recording && setKernelActivity(want))
    recording = want;
}

// Whether s has profiling on at time now, and when that next changes
// (UINT64_MAX if never, absent a settings change).
static bool gateAt(const LiveSettings &s, uint64_t now, uint64_t &next) {
  next = UINT64_MAX;
  if (!s.enabled)
    return false;
  if (s.dutyCycle >= 1.0)
    return true;
  uint64_t period = (uint64_t)(s.dutyPeriodSec * kNsPerSec);
  uint64_t onNs = (uint64_t)(s.dutyCycle * (double)period);
  uint64_t phase = now % period;
  bool on = phase < onNs;
  next = now - phase + (on ? onNs : period);
  return on;
}

static void sleepUntil(uint64_t ns) {
  struct timespec ts = {(time_t)(ns / kNsPerSec), (long)(ns % kNsPerSec)};
  while (clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts, nullptr) ==
         EINTR) {
  }
}

// One pass of the config thread: re-read the file if due, then update the
// gate and kernel activity recording. Returns when the next pass is due.
static uint64_t configStep(ConfigThread &t, uint64_t &nextFileCheck) {
  uint64_t now = realtimeNs();
  if (t.path && now >= nextFileCheck) {
    if (reloadIfChanged(t))
      applyLiveSettings(t.cur);
    nextFileCheck = now + kFileCheckNs;
  }
  uint64_t next;
  setActive(gateAt(t.cur, now, next));
  updateKernelActivity();
  next = std::min(next, now + kAttachPollNs);
  if (t.path)
    next = std::min(next, nextFileCheck);
  return next;
}

static void *configThreadMain(void *arg) {
  auto &t = *static_cast<ConfigThread *>(arg);
  uint64_t nextFileCheck = realtimeNs() + kFileCheckNs;
  for (;;)
    sleepUntil(configStep(t, nextFileCheck));
}

void startRuntimeConfig() {
  static ConfigThread t; // lives as long as the thread; never destroyed early
  t.path = getenv("PARCAGPU_CONFIG_FILE");
  if (t.path && *t.path == '\0')
    t.path = nullptr;
  t.env = envSettings();
  t.cur = t.env;
  logSettings("env", t.cur);

  // Settle the gate and kernel recording before the first kernel runs.
  uint64_t nextFileCheck = 0;
  configStep(t, nextFileCheck);

  // Block signals in the thread so the host process's handlers never run on
  // it.
  sigset_t all, old;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &old);
  pthread_t tid;
  int rc = pthread_create(&tid, nullptr, configThreadMain, &t);
  pthread_sigmask(SIG_SETMASK, &old, nullptr);
  if (rc != 0) {
    DEBUG_PRINTF("[PARCAGPU] can't start config thread: %s\n", strerror(rc));
    fireError(rc, "can't start config thread", "runtime_config");
    return;
  }
  pthread_setname_np(tid, "parcagpu-config");
  pthread_detach(tid);
}

} // namespace parcagpu
