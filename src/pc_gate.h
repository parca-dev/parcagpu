// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PC_GATE_H_
#define PC_GATE_H_

#include <cstdint>

#include <cuda.h>
#include <cupti.h>

namespace parcagpu::gate {

// The PC sampling gate. Every driver API call passes through it between its
// ENTER and EXIT callbacks, except calls that only wait, query or read
// context state ("exempt"). It guards against two hangs:
//
// - cuptiPCSamplingEnable/Disable/Start/Stop deadlock against another thread
//   inside a kernel launch, memset or memcpy: with PC sampling enabled, CUPTI
//   hooks those calls and takes its own mutex under the driver lock, while
//   the PC sampling call holds that mutex and waits for the driver lock
//   (test/relrace.cu reproduces it). So a state change runs only with the
//   gate closed (Closed): no other thread inside a gated call, new ones
//   waiting at ENTER.
// - While a KERNEL_SERIALIZED window is open, the launching thread holds the
//   driver lock and other threads' calls starve; a hang if the window's
//   owner then waits for one of them, or is starved itself and never closes
//   the window. So while a window is open (own) only its owner's gated calls
//   are admitted; the others wait at ENTER, outside the driver.
//
// Nothing waits at the gate without a bound. A call waiting for a window
// that outlived its deadline closes it itself (the overdue handler), and if
// that fails (its owner is inside a driver call) stops honoring it: the gate
// fails open, the window stays open until its owner closes it. A call
// waiting for a close proceeds after kEnterWaitNs.

// Default time to wait for the gate to close. A caller that gives up retries
// later; meanwhile the application's gated calls wait up to this long.
constexpr uint64_t kCloseTimeoutNs = 100'000'000ULL;
// The longest a call waits at ENTER for a close; then it goes in regardless.
// Longer than a close may take (kCloseTimeoutNs), so that it only goes in
// on a close that's stuck.
constexpr uint64_t kEnterWaitNs = 250'000'000ULL;
// A window's deadline is its length plus this much.
constexpr uint64_t kWindowSlackNs = 20'000'000ULL;
// How long the overdue handler tries to close the gate: if the window's
// owner is still inside a call by then, the gate fails open instead.
constexpr uint64_t kOverdueCloseTimeoutNs = 10'000'000ULL;

// Turns the gate on: from here on enter()/exit() count. Call before the first
// driver API callback is enabled, so that no call is in flight uncounted.
void enable();
bool enabled();

// Closes the window on ctx (stops sampling, which disowns it), closing the
// gate for at most kOverdueCloseTimeoutNs, without backoff; true if it did.
// Called from a waiting thread at its ENTER, with no gate lock held.
using OverdueHandler = bool (*)(CUcontext ctx);
void setOverdueHandler(OverdueHandler fn);

// The driver API callback's ENTER and EXIT. internal: a call parcagpu makes
// itself, which doesn't wait for windows (it may hold parcagpu's locks).
void enter(CUpti_CallbackId cbid, bool internal = false);
void exit(CUpti_CallbackId cbid);

// The calling thread owns a window on ctx for the next windowNs (plus
// kWindowSlackNs): other threads' gated calls wait until it is disowned.
// Call with the gate closed, right after the window opens.
void own(CUcontext ctx, uint64_t windowNs);
// The window on ctx closed (any thread).
void disown(CUcontext ctx);

// Closes the gate for its lifetime: ok() once no other thread is inside a
// gated call, false if that took longer than timeoutNs. Nests on the closing
// thread, whose own driver calls pass. A no-op (ok) when the gate is off or
// this thread bypasses it. A timeout makes closes fail at once for a while
// (a call is stuck); with backoff false, a close neither waits out nor
// causes that.
class Closed {
public:
  explicit Closed(uint64_t timeoutNs = kCloseTimeoutNs, bool backoff = true);
  ~Closed();
  Closed(const Closed &) = delete;
  Closed &operator=(const Closed &) = delete;
  bool ok() const { return ok_; }

private:
  bool ok_ = true;
  bool owner_ = false;
  bool nested_ = false;
};

// For its lifetime this thread's PC sampling calls don't close the gate: in
// the context created/destroy callbacks, which run inside the driver and
// where the context isn't in use.
class Bypass {
public:
  Bypass();
  ~Bypass();
  Bypass(const Bypass &) = delete;
  Bypass &operator=(const Bypass &) = delete;

private:
  bool prev_;
};

// Counters, for PARCAGPU_GATE_STATS and tests.
struct Stats {
  uint64_t closes = 0;
  uint64_t timeouts = 0;
  uint64_t waitNs = 0;    // total time spent closing
  uint64_t maxWaitNs = 0; // longest close
  uint64_t maxHeldNs = 0; // longest the gate stayed closed
  uint64_t windowWaits = 0;     // calls that waited for a window
  uint64_t maxWindowWaitNs = 0; // longest such wait
  uint64_t overdueCloses = 0;   // overdue windows closed by a waiting call
  uint64_t windowFailOpens = 0; // overdue windows no longer honored
  uint64_t enterFailOpens = 0;  // calls that stopped waiting for a close
};
Stats stats();

} // namespace parcagpu::gate

#endif // PC_GATE_H_
