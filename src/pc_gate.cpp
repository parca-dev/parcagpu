// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include "pc_gate.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <time.h>
#include <vector>

namespace parcagpu::gate {

namespace {

// After a close times out, closes fail at once for this long, so that a gate
// that can't close (a thread stuck in a long driver call) doesn't stall the
// application on every attempt.
constexpr uint64_t kBackoffNs = 2'000'000'000ULL;
constexpr uint64_t kStatsPeriodNs = 10'000'000'000ULL;

std::atomic<bool> g_enabled{false};
bool g_printStats = false;
std::atomic<OverdueHandler> g_overdue{nullptr};

// Threads inside a counted driver call, and whether a closer wants them out.
// enter() bumps g_inside then reads g_closing, a closer sets g_closing then
// reads g_inside (both seq_cst): one of them sees the other.
std::atomic<int> g_inside{0};
std::atomic<bool> g_closing{false};
std::mutex g_mutex;
std::condition_variable g_cv;
std::atomic<uint64_t> g_backoffUntilNs{0};

// Open windows, under g_mutex. g_honored counts those not failed open: the
// fast path's check. own() runs with the gate closed, so a call that enters
// after the close either saw g_closing or sees g_honored (seq_cst).
struct Window {
  uint64_t id;
  CUcontext ctx;
  std::thread::id owner;
  uint64_t deadlineNs;
  bool failedOpen = false;
  bool closing = false; // a waiting call is closing it (overdue)
};
std::vector<Window> g_windows;
std::atomic<int> g_honored{0};
uint64_t g_nextWindowId = 1;

thread_local int t_depth = 0;        // driver calls this thread is inside
thread_local bool t_counted = false; // its outermost one is in g_inside
thread_local int t_closed = 0;       // Closed nesting on this thread
thread_local bool t_parked = false;  // left g_inside to close the gate
thread_local bool t_bypass = false;

std::atomic<uint64_t> g_closes{0}, g_timeouts{0}, g_waitNs{0}, g_maxWaitNs{0},
    g_maxHeldNs{0}, g_windowWaits{0}, g_maxWindowWaitNs{0},
    g_overdueCloses{0}, g_windowFailOpens{0}, g_enterFailOpens{0};
uint64_t g_closedAtNs = 0;    // the owner's, while closed
uint64_t g_lastStatsNs = 0;   // under g_mutex

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void storeMax(std::atomic<uint64_t> &a, uint64_t v) {
  uint64_t cur = a.load(std::memory_order_relaxed);
  while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed))
    ;
}

// Calls that only wait for the GPU, query, or read or switch the current
// context: they don't reach PC sampling's hooks, and a thread in a long wait
// mustn't hold the gate open.
bool isExempt(CUpti_CallbackId cbid) {
  switch (cbid) {
  case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
  case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamQuery:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamQuery_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuEventQuery:
  case CUPTI_DRIVER_TRACE_CBID_cuCtxGetCurrent:
  case CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent:
  case CUPTI_DRIVER_TRACE_CBID_cuCtxPushCurrent_v2:
  case CUPTI_DRIVER_TRACE_CBID_cuCtxPopCurrent_v2:
  case CUPTI_DRIVER_TRACE_CBID_cuCtxGetDevice:
  case CUPTI_DRIVER_TRACE_CBID_cuCtxGetId:
  case CUPTI_DRIVER_TRACE_CBID_cuDeviceGetAttribute:
  case CUPTI_DRIVER_TRACE_CBID_cuDriverGetVersion:
  case CUPTI_DRIVER_TRACE_CBID_cuGetProcAddress:
  case CUPTI_DRIVER_TRACE_CBID_cuGetProcAddress_v2:
  case CUPTI_DRIVER_TRACE_CBID_cuPointerGetAttribute:
  case CUPTI_DRIVER_TRACE_CBID_cuPointerGetAttributes:
    return true;
  default:
    return false;
  }
}

void maybePrintStatsLocked(uint64_t now) {
  if (!g_printStats || now - g_lastStatsNs < kStatsPeriodNs)
    return;
  g_lastStatsNs = now;
  fprintf(stderr,
          "[PARCAGPU] gate: closes=%lu timeouts=%lu wait_ms=%.1f "
          "max_wait_ms=%.1f max_held_ms=%.1f window_waits=%lu "
          "max_window_wait_ms=%.1f overdue_closes=%lu window_fail_opens=%lu "
          "enter_fail_opens=%lu\n",
          g_closes.load(), g_timeouts.load(), g_waitNs.load() / 1e6,
          g_maxWaitNs.load() / 1e6, g_maxHeldNs.load() / 1e6,
          g_windowWaits.load(), g_maxWindowWaitNs.load() / 1e6,
          g_overdueCloses.load(), g_windowFailOpens.load(),
          g_enterFailOpens.load());
}

Window *findWindowLocked(uint64_t id) {
  for (auto &w : g_windows)
    if (w.id == id)
      return &w;
  return nullptr;
}

void failOpenLocked(Window &w) {
  if (w.failedOpen)
    return;
  w.failedOpen = true;
  g_honored.fetch_sub(1);
  g_windowFailOpens.fetch_add(1, std::memory_order_relaxed);
  g_cv.notify_all();
}

// enter()'s slow path: a close is under way or a window is open. Called with
// this call counted in g_inside; returns with it counted again.
void waitAtGate(bool internal) {
  std::unique_lock<std::mutex> lock(g_mutex);
  const auto self = std::this_thread::get_id();
  auto blocking = [&]() -> Window * {
    if (internal)
      return nullptr;
    for (auto &w : g_windows)
      if (!w.failedOpen && w.owner != self)
        return &w;
    return nullptr;
  };
  if (!g_closing.load() && !blocking())
    return;
  // Step back out and wait at the gate.
  g_inside.fetch_sub(1);
  g_cv.notify_all();
  const uint64_t start = nowNs();
  bool windowWait = false;
  for (;;) {
    const uint64_t now = nowNs();
    if (g_closing.load()) {
      if (now - start >= kEnterWaitNs) {
        g_enterFailOpens.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      g_cv.wait_for(lock, std::chrono::nanoseconds(kEnterWaitNs - (now - start)));
      continue;
    }
    Window *w = blocking();
    if (!w)
      break;
    windowWait = true;
    if (now < w->deadlineNs) {
      g_cv.wait_for(lock, std::chrono::nanoseconds(w->deadlineNs - now));
      continue;
    }
    // Overdue: its owner went quiet, or is stuck in the driver.
    if (w->closing) {
      // Another waiting call is closing it; it won't take long, but bound it.
      if (now - w->deadlineNs >= kEnterWaitNs) {
        failOpenLocked(*w);
        continue;
      }
      g_cv.wait_for(lock, std::chrono::milliseconds(1));
      continue;
    }
    const uint64_t id = w->id;
    const CUcontext ctx = w->ctx;
    OverdueHandler handler = g_overdue.load();
    bool closed = false;
    if (handler) {
      w->closing = true;
      lock.unlock();
      closed = handler(ctx); // closes the gate, stops, disowns
      lock.lock();
    }
    if (closed)
      g_overdueCloses.fetch_add(1, std::memory_order_relaxed);
    // Still here: its owner is inside a driver call (or there's no handler).
    // Stop honoring it; its owner closes it at its next launch EXIT.
    if (Window *still = findWindowLocked(id)) {
      still->closing = false;
      failOpenLocked(*still);
    }
  }
  if (windowWait) {
    g_windowWaits.fetch_add(1, std::memory_order_relaxed);
    storeMax(g_maxWindowWaitNs, nowNs() - start);
  }
  g_inside.fetch_add(1);
}

} // namespace

void enable() {
  g_printStats = getenv("PARCAGPU_GATE_STATS") != nullptr;
  g_enabled.store(true, std::memory_order_release);
}

bool enabled() { return g_enabled.load(std::memory_order_acquire); }

void setOverdueHandler(OverdueHandler fn) { g_overdue.store(fn); }

void enter(CUpti_CallbackId cbid, bool internal) {
  if (!g_enabled.load(std::memory_order_relaxed))
    return;
  if (t_depth++ > 0 || t_closed > 0 || isExempt(cbid))
    return;
  g_inside.fetch_add(1);
  if (g_closing.load() || (!internal && g_honored.load() > 0))
    waitAtGate(internal);
  t_counted = true;
}

void exit(CUpti_CallbackId) {
  if (!g_enabled.load(std::memory_order_relaxed) || t_depth == 0)
    return;
  if (--t_depth > 0 || !t_counted)
    return;
  t_counted = false;
  g_inside.fetch_sub(1);
  if (g_closing.load()) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_cv.notify_all();
  }
}

void own(CUcontext ctx, uint64_t windowNs) {
  if (!g_enabled.load(std::memory_order_acquire))
    return;
  std::lock_guard<std::mutex> lock(g_mutex);
  Window w;
  w.id = g_nextWindowId++;
  w.ctx = ctx;
  w.owner = std::this_thread::get_id();
  w.deadlineNs = nowNs() + windowNs + kWindowSlackNs;
  g_windows.push_back(w);
  g_honored.fetch_add(1);
}

void disown(CUcontext ctx) {
  if (!g_enabled.load(std::memory_order_acquire))
    return;
  std::lock_guard<std::mutex> lock(g_mutex);
  for (auto it = g_windows.begin(); it != g_windows.end();) {
    if (it->ctx != ctx) {
      ++it;
      continue;
    }
    if (!it->failedOpen)
      g_honored.fetch_sub(1);
    it = g_windows.erase(it);
  }
  g_cv.notify_all();
}

Closed::Closed(uint64_t timeoutNs, bool backoff) {
  if (!g_enabled.load(std::memory_order_acquire) || t_bypass)
    return;
  if (t_closed > 0) {
    t_closed++; // nested: already closed by this thread
    nested_ = true;
    return;
  }
  const uint64_t start = nowNs();
  if (backoff && start < g_backoffUntilNs.load(std::memory_order_relaxed)) {
    ok_ = false;
    return;
  }
  std::unique_lock<std::mutex> lock(g_mutex);
  // A caller inside a driver call (a CUPTI callback) steps out of g_inside
  // while it closes: it isn't in the driver proper, and another closer must
  // not wait for it.
  bool parked = t_counted;
  if (parked) {
    t_counted = false;
    g_inside.fetch_sub(1);
    g_cv.notify_all();
  }
  t_closed = 1; // our own driver calls from here on aren't counted
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeoutNs);
  bool got = g_cv.wait_until(lock, deadline, [] { return !g_closing.load(); });
  if (got) {
    g_closing.store(true);
    got = g_cv.wait_until(lock, deadline,
                          [] { return g_inside.load() == 0; });
    if (!got) {
      g_closing.store(false);
      g_cv.notify_all();
    }
  }
  const uint64_t waited = nowNs() - start;
  if (!got) {
    t_closed = 0;
    if (parked) {
      // Back in as before, once any other closer is done (bounded, as at
      // ENTER).
      if (!g_cv.wait_for(lock, std::chrono::nanoseconds(kEnterWaitNs),
                         [] { return !g_closing.load(); }))
        g_enterFailOpens.fetch_add(1, std::memory_order_relaxed);
      g_inside.fetch_add(1);
      t_counted = true;
    }
    g_timeouts.fetch_add(1, std::memory_order_relaxed);
    if (backoff)
      g_backoffUntilNs.store(nowNs() + kBackoffNs, std::memory_order_relaxed);
    maybePrintStatsLocked(nowNs());
    ok_ = false;
    return;
  }
  t_parked = parked;
  owner_ = true;
  g_closedAtNs = nowNs();
  g_closes.fetch_add(1, std::memory_order_relaxed);
  g_waitNs.fetch_add(waited, std::memory_order_relaxed);
  storeMax(g_maxWaitNs, waited);
}

Closed::~Closed() {
  if (nested_)
    t_closed--;
  if (!owner_)
    return;
  std::lock_guard<std::mutex> lock(g_mutex);
  const uint64_t now = nowNs();
  storeMax(g_maxHeldNs, now - g_closedAtNs);
  g_closing.store(false);
  t_closed = 0;
  if (t_parked) {
    t_parked = false;
    g_inside.fetch_add(1);
    t_counted = true;
  }
  g_cv.notify_all();
  maybePrintStatsLocked(now);
}

Bypass::Bypass() : prev_(t_bypass) { t_bypass = true; }
Bypass::~Bypass() { t_bypass = prev_; }

Stats stats() {
  Stats s;
  s.closes = g_closes.load();
  s.timeouts = g_timeouts.load();
  s.waitNs = g_waitNs.load();
  s.maxWaitNs = g_maxWaitNs.load();
  s.maxHeldNs = g_maxHeldNs.load();
  s.windowWaits = g_windowWaits.load();
  s.maxWindowWaitNs = g_maxWindowWaitNs.load();
  s.overdueCloses = g_overdueCloses.load();
  s.windowFailOpens = g_windowFailOpens.load();
  s.enterFailOpens = g_enterFailOpens.load();
  return s;
}

} // namespace parcagpu::gate
