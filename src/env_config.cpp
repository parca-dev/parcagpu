// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include "env_config.h"
#include "log.h"
#include "pc_sampling.h" // kMaxPCHardwareBufferMB

#include <cstdlib>
#include <cstring>

extern char **environ;

namespace parcagpu {

static constexpr const char *kComponent = "env_config";

// Known PARCAGPU_* environment variables.
static const char *knownVars[] = {
    "PARCAGPU_DEBUG",
    "PARCAGPU_LOG_LEVEL",
    "PARCAGPU_RATE_LIMIT",
    "PARCAGPU_GRAPH_RATE_LIMIT",
    "PARCAGPU_SAMPLING_FACTOR",
    "PARCAGPU_PC_SAMPLING_RATE",
    "PARCAGPU_PC_HW_BUFFER_MB",
    "PARCAGPU_GATE_STATS",
    "PARCAGPU_METADATA_REFRESH_MS",
    "PARCAGPU_BLOCKING_CALLS",
    "PARCAGPU_BLOCKING_THRESHOLD_US",
};
static constexpr size_t numKnownVars =
    sizeof(knownVars) / sizeof(knownVars[0]);

static bool isKnown(const char *name, size_t nameLen) {
  for (size_t i = 0; i < numKnownVars; ++i) {
    if (std::strlen(knownVars[i]) == nameLen &&
        std::strncmp(knownVars[i], name, nameLen) == 0)
      return true;
  }
  return false;
}

void validateEnvVars() {
  // Scan environment for unrecognized PARCAGPU_* variables.
  static constexpr const char prefix[] = "PARCAGPU_";
  static constexpr size_t prefixLen = sizeof(prefix) - 1;

  for (char **ep = environ; *ep; ++ep) {
    if (std::strncmp(*ep, prefix, prefixLen) != 0)
      continue;

    // Extract variable name (everything before '=').
    const char *eq = std::strchr(*ep, '=');
    size_t nameLen = eq ? (size_t)(eq - *ep) : std::strlen(*ep);

    if (!isKnown(*ep, nameLen)) {
      // Null-terminate for printing.
      char nameBuf[128] = {};
      size_t copyLen = nameLen < sizeof(nameBuf) - 1 ? nameLen : sizeof(nameBuf) - 1;
      std::memcpy(nameBuf, *ep, copyLen);

      PLOG_WARN(0, kComponent, "unrecognized env var '%s'", nameBuf);
    }
  }

  // Validate specific variables.
  const char *val;

  val = std::getenv("PARCAGPU_LOG_LEVEL");
  if (val && !log::parseLevel(val)) {
    PLOG_WARN(0, kComponent,
              "PARCAGPU_LOG_LEVEL=%s invalid (debug, info, warn or error), "
              "using info",
              val);
  }

  val = std::getenv("PARCAGPU_RATE_LIMIT");
  if (val) {
    double rate = std::atof(val);
    if (rate <= 0) {
      PLOG_WARN(0, kComponent,
                "PARCAGPU_RATE_LIMIT=%s invalid "
                "(must be > 0), using default",
                val);
    }
  }

  val = std::getenv("PARCAGPU_GRAPH_RATE_LIMIT");
  if (val) {
    double rate = std::atof(val);
    if (rate <= 0) {
      PLOG_WARN(0, kComponent,
                "PARCAGPU_GRAPH_RATE_LIMIT=%s invalid "
                "(must be > 0), using default",
                val);
    }
  }

  val = std::getenv("PARCAGPU_SAMPLING_FACTOR");
  if (val) {
    int factor = std::atoi(val);
    if (factor != 0 && (factor < 5 || factor > 31)) {
      PLOG_WARN(0, kComponent,
                "PARCAGPU_SAMPLING_FACTOR=%s out of "
                "range [0, 5-31], using default",
                val);
    }
  }

  val = std::getenv("PARCAGPU_PC_SAMPLING_RATE");
  if (val) {
    double r = std::atof(val);
    if (r < 0.0) {
      PLOG_WARN(0, kComponent,
                "PARCAGPU_PC_SAMPLING_RATE=%s "
                "invalid (must be >= 0), using default",
                val);
    }
  }

  val = std::getenv("PARCAGPU_METADATA_REFRESH_MS");
  if (val && std::atol(val) <= 0) {
    PLOG_WARN(0, kComponent,
              "PARCAGPU_METADATA_REFRESH_MS=%s invalid "
              "(must be > 0), using default",
              val);
  }

  val = std::getenv("PARCAGPU_PC_HW_BUFFER_MB");
  if (val) {
    long mb = std::atol(val);
    if (mb < 0 || mb > (long)kMaxPCHardwareBufferMB) {
      PLOG_WARN(0, kComponent,
                "PARCAGPU_PC_HW_BUFFER_MB=%s invalid "
                "(must be 0-%u), ignoring",
                val, kMaxPCHardwareBufferMB);
    }
  }

  val = std::getenv("PARCAGPU_BLOCKING_THRESHOLD_US");
  if (val) {
    double us = std::atof(val);
    if (us < 0.0) {
      PLOG_WARN(0, kComponent,
                "PARCAGPU_BLOCKING_THRESHOLD_US=%s "
                "invalid (must be >= 0), using default",
                val);
    }
  }
}

} // namespace parcagpu
