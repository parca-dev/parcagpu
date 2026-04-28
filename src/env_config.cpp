// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include "env_config.h"
#include "pc_sampling.h" // DEBUG_PRINTF, fireError
#include "probes.h"

#include <cstdlib>
#include <cstring>

extern char **environ;

namespace parcagpu {

// Known PARCAGPU_* environment variables.
static const char *knownVars[] = {
    "PARCAGPU_DEBUG",
    "PARCAGPU_RATE_LIMIT",
    "PARCAGPU_SAMPLING_FACTOR",
    "PARCAGPU_PC_SAMPLING_PROBABILITY",
    "PARCAGPU_PC_SAMPLING_INTERVAL",
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

      DEBUG_PRINTF("[PARCAGPU] Warning: unrecognized env var '%s'\n", nameBuf);
      fireError(0, nameBuf, "env_config: unrecognized variable");
    }
  }

  // Validate specific variables.
  const char *val;

  val = std::getenv("PARCAGPU_RATE_LIMIT");
  if (val) {
    double rate = std::atof(val);
    if (rate <= 0) {
      DEBUG_PRINTF("[PARCAGPU] Warning: PARCAGPU_RATE_LIMIT=%s invalid "
                   "(must be > 0), using default\n", val);
      fireError(0, val, "env_config: PARCAGPU_RATE_LIMIT invalid");
    }
  }

  val = std::getenv("PARCAGPU_SAMPLING_FACTOR");
  if (val) {
    int factor = std::atoi(val);
    if (factor != 0 && (factor < 5 || factor > 31)) {
      DEBUG_PRINTF("[PARCAGPU] Warning: PARCAGPU_SAMPLING_FACTOR=%s out of "
                   "range [0, 5-31], using default\n", val);
      fireError(0, val, "env_config: PARCAGPU_SAMPLING_FACTOR out of range");
    }
  }

  val = std::getenv("PARCAGPU_PC_SAMPLING_PROBABILITY");
  if (val) {
    double p = std::atof(val);
    if (p <= 0.0 || p > 1.0) {
      DEBUG_PRINTF("[PARCAGPU] Warning: PARCAGPU_PC_SAMPLING_PROBABILITY=%s "
                   "invalid (must be in (0, 1]), using default\n", val);
      fireError(0, val,
                "env_config: PARCAGPU_PC_SAMPLING_PROBABILITY invalid");
    }
  }

  val = std::getenv("PARCAGPU_PC_SAMPLING_INTERVAL");
  if (val) {
    double s = std::atof(val);
    if (s <= 0.0) {
      DEBUG_PRINTF("[PARCAGPU] Warning: PARCAGPU_PC_SAMPLING_INTERVAL=%s "
                   "invalid (must be > 0), using default\n", val);
      fireError(0, val,
                "env_config: PARCAGPU_PC_SAMPLING_INTERVAL invalid");
    }
  }
}

} // namespace parcagpu
