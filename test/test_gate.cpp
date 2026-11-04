// test_gate.cpp — unit test for the PC sampling gate (src/pc_gate.cpp), no
// CUDA needed. Threads simulate driver API calls: the gate's ENTER callback,
// a call body, the EXIT callback; some close the gate from their callbacks
// as parcagpu does on launches, and a release thread closes it from outside.
// Checks that no call body runs while the gate is closed, that nothing
// deadlocks, that a stuck call makes closers time out rather than hang, and
// that synchronize calls don't hold the gate. Then windows: while one is
// open only its owner's calls run, and a window whose owner goes quiet or
// gets stuck in a call holds the others back no longer than its deadline.
//
// Build and run: make test-gate (with ThreadSanitizer).

#include "pc_gate.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

namespace gate = parcagpu::gate;
using namespace std::chrono_literals;

namespace {

const CUpti_CallbackId kLaunch = CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel;
const CUpti_CallbackId kMemcpy = CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2;
const CUpti_CallbackId kSync = CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize;

std::atomic<int> g_inBody{0}; // threads inside a gated call body
std::atomic<int> g_violations{0};
std::atomic<uint64_t> g_calls{0}, g_closedOk{0}, g_closedFail{0};
std::atomic<bool> g_stop{false};

// With the gate closed by this thread: nobody may be in a call body.
void closedSection() {
  if (g_inBody.load() != 0)
    g_violations++;
  std::this_thread::sleep_for(20us);
  if (g_inBody.load() != 0)
    g_violations++;
}

void tryClose() {
  gate::Closed c;
  if (c.ok()) {
    g_closedOk++;
    closedSection();
  } else {
    g_closedFail++;
  }
}

// One simulated driver call. closeAt: 0 none, 1 in ENTER, 2 in EXIT.
void call(CUpti_CallbackId cbid, int closeAt, bool nested, int bodyUs) {
  gate::enter(cbid);
  if (closeAt == 1)
    tryClose();
  bool counted = cbid != kSync;
  if (counted)
    g_inBody++;
  if (nested) { // a driver call made by this one (runtime -> driver, etc.)
    gate::enter(kMemcpy);
    gate::exit(kMemcpy);
  }
  if (bodyUs)
    std::this_thread::sleep_for(std::chrono::microseconds(bodyUs));
  if (counted)
    g_inBody--;
  if (closeAt == 2)
    tryClose();
  gate::exit(cbid);
  g_calls++;
}

void appThread(int seed) {
  std::mt19937 rng(seed);
  while (!g_stop.load()) {
    int r = rng() % 1000;
    if (r < 5)
      call(kLaunch, 1, false, 5); // a window opens at a launch ENTER
    else if (r < 10)
      call(kLaunch, 2, false, 5); // or closes at its EXIT
    else if (r < 300)
      call(kMemcpy, 0, r % 2, 2);
    else if (r < 310)
      call(kSync, 0, false, 300); // long wait for the GPU
    else
      call(kLaunch, 0, r % 3 == 0, 1);
  }
}

void releaseThread() {
  while (!g_stop.load()) {
    tryClose();
    {
      gate::Closed outer; // nests
      if (outer.ok()) {
        gate::Closed inner;
        if (!inner.ok())
          g_violations++;
        closedSection();
        // The closer's own driver calls pass.
        gate::enter(kLaunch);
        gate::exit(kLaunch);
      }
    }
    std::this_thread::sleep_for(500us);
  }
}

bool run(const char *name, int apps, std::chrono::milliseconds dur,
         bool stuck) {
  g_stop = false;
  g_violations = 0;
  g_calls = g_closedOk = g_closedFail = 0;
  auto before = gate::stats();
  std::vector<std::thread> ts;
  for (int i = 0; i < apps; i++)
    ts.emplace_back(appThread, i + 1);
  ts.emplace_back(releaseThread);
  std::thread stuckThread;
  if (stuck) // a call that stays in the driver longer than closers wait
    stuckThread = std::thread(
        [] { call(kMemcpy, 0, false, (int)(3 * gate::kCloseTimeoutNs / 1000)); });
  std::this_thread::sleep_for(dur);
  g_stop = true;
  for (auto &t : ts)
    t.join();
  if (stuckThread.joinable())
    stuckThread.join();
  auto after = gate::stats();
  uint64_t timeouts = after.timeouts - before.timeouts;
  bool ok = g_violations == 0 && g_closedOk > 0 && (!stuck || timeouts > 0);
  printf("%-28s %s: calls=%lu closes=%lu failed=%lu timeouts=%lu "
         "violations=%d max_wait_ms=%.1f max_held_ms=%.1f\n",
         name, ok ? "ok" : "FAIL", g_calls.load(), g_closedOk.load(),
         g_closedFail.load(), timeouts, g_violations.load(),
         after.maxWaitNs / 1e6, after.maxHeldNs / 1e6);
  return ok;
}

// Windows. A fake context; the overdue handler closes the window as
// parcagpu's does (gate closed, then disown), unless its owner is inside a
// call.
CUcontext const kCtx = reinterpret_cast<CUcontext>(0x1000);
std::atomic<bool> g_windowOpen{false}; // honored: only the owner may run
std::atomic<uint64_t> g_otherCalls{0};

bool closeOverdue(CUcontext ctx) {
  gate::Closed c(gate::kOverdueCloseTimeoutNs, false);
  if (!c.ok())
    return false;
  g_windowOpen = false;
  gate::disown(ctx);
  return true;
}

void otherThread() {
  while (!g_stop.load()) {
    gate::enter(kMemcpy);
    if (g_windowOpen.load())
      g_violations++;
    gate::exit(kMemcpy);
    g_otherCalls++;
  }
}

// owner: 0 closes its window itself, 1 goes quiet, 2 gets stuck in a call.
bool runWindow(const char *name, int owner) {
  g_stop = false;
  g_violations = 0;
  g_otherCalls = 0;
  auto before = gate::stats();
  std::vector<std::thread> ts;
  for (int i = 0; i < 3; i++)
    ts.emplace_back(otherThread);
  std::thread o([owner] {
    for (int w = 0; w < 5; w++) {
      std::this_thread::sleep_for(20ms);
      {
        gate::Closed c;
        if (!c.ok())
          continue;
        gate::own(kCtx, 10'000'000ULL);
        g_windowOpen = owner != 2; // a stuck owner's window fails open
      }
      auto end = std::chrono::steady_clock::now() + 10ms;
      while (std::chrono::steady_clock::now() < end) { // its own calls pass
        gate::enter(kLaunch);
        gate::exit(kLaunch);
      }
      if (owner == 1) {
        std::this_thread::sleep_for(100ms); // quiet: overdue at 30 ms
      } else if (owner == 2) {
        gate::enter(kLaunch); // stuck in a call past the deadline
        std::this_thread::sleep_for(100ms);
        gate::exit(kLaunch);
      }
      gate::Closed c;
      g_windowOpen = false;
      gate::disown(kCtx); // a no-op if it was closed for it
    }
  });
  o.join();
  g_stop = true;
  for (auto &t : ts)
    t.join();
  auto after = gate::stats();
  uint64_t overdue = after.overdueCloses - before.overdueCloses;
  uint64_t failOpens = after.windowFailOpens - before.windowFailOpens;
  bool ok = g_violations == 0 && g_otherCalls > 0 &&
            after.maxWindowWaitNs < 80'000'000ULL &&
            (owner != 1 || overdue > 0) && (owner != 2 || failOpens > 0);
  printf("%-28s %s: other_calls=%lu violations=%d window_waits=%lu "
         "max_window_wait_ms=%.1f overdue_closes=%lu fail_opens=%lu\n",
         name, ok ? "ok" : "FAIL", g_otherCalls.load(), g_violations.load(),
         after.windowWaits - before.windowWaits, after.maxWindowWaitNs / 1e6,
         overdue, failOpens);
  return ok;
}

} // namespace

int main() {
  gate::enable();
  bool ok = true;
  ok &= run("4 app threads", 4, 2000ms, false);
  ok &= run("16 app threads", 16, 2000ms, false);
  // A stuck call: closes time out (then back off) instead of hanging.
  ok &= run("stuck call", 4, 1500ms, true);
  // After the backoff, the gate closes again.
  std::this_thread::sleep_for(2100ms);
  ok &= run("after stuck call", 4, 1000ms, false);
  gate::setOverdueHandler(closeOverdue);
  ok &= runWindow("window, owner closes", 0);
  ok &= runWindow("window, owner quiet", 1);
  ok &= runWindow("window, owner stuck", 2);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
