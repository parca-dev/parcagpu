// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_TOKEN_BUCKET_H_
#define PARCAGPU_TOKEN_BUCKET_H_

#include <cstdint>
#include <time.h>

namespace parcagpu {

// Simple token bucket rate limiter. Not thread-safe — use one instance per
// thread, or a thread_local instance when shared across call sites.
class TokenBucket {
public:
  // startFull=true: first tryAcquire succeeds immediately.
  // startFull=false: must wait for refill before first success.
  explicit TokenBucket(double tokensPerSec, bool startFull = true)
      : rate(tokensPerSec), tokens(startFull ? 1.0 : 0.0) {}

  void setRate(double tokensPerSec) {
    rate = tokensPerSec;
    if (tokens > rate)
      tokens = rate;
  }

  // Returns true if a token was available and consumed.
  bool tryAcquire() {
    refill();
    if (tokens >= 1.0) {
      tokens -= 1.0;
      return true;
    }
    return false;
  }

private:
  void refill() {
    uint64_t now = nowNs();
    if (lastRefillNs > 0) {
      double elapsed = (now - lastRefillNs) / 1e9;
      tokens += elapsed * rate;
      if (tokens > rate)
        tokens = rate;
    }
    lastRefillNs = now;
  }

  static uint64_t nowNs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
  }

  double rate;
  double tokens;
  uint64_t lastRefillNs = 0;
};

} // namespace parcagpu

#endif // PARCAGPU_TOKEN_BUCKET_H_
