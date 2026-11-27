// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_PROFILING_ACCESS_H_
#define PARCAGPU_PROFILING_ACCESS_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace parcagpu {

// Whether the NVIDIA driver lets this process use the GPU performance
// counters that PC sampling needs. With NVreg_RestrictProfilingToAdminUsers=1
// (the default, shown as "RmProfilingAdminOnly: 1" in the driver's params)
// only processes with CAP_SYS_ADMIN may, whatever their uid. CUPTI doesn't
// say so up front: cuptiPCSamplingEnable and the configuration calls succeed,
// and at the first module load or kernel launch CUPTI fails with
// CUPTI_ERROR_INSUFFICIENT_PRIVILEGES and finalizes itself, which drops
// parcagpu's subscriber and every activity kind along with PC sampling.
// True when the driver doesn't say (no params file, or no such line).
inline bool profilingAllowed(const char *params = "/proc/driver/nvidia/params",
                             const char *status = "/proc/self/status") {
  FILE *f = fopen(params, "r");
  if (!f)
    return true;
  char line[256];
  int adminOnly = 0;
  while (fgets(line, sizeof(line), f)) {
    if (sscanf(line, "RmProfilingAdminOnly: %d", &adminOnly) == 1)
      break;
  }
  fclose(f);
  if (adminOnly == 0)
    return true;

  f = fopen(status, "r");
  if (!f)
    return false;
  unsigned long long capEff = 0;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "CapEff:", 7) == 0) {
      capEff = strtoull(line + 7, nullptr, 16);
      break;
    }
  }
  fclose(f);
  constexpr int kCapSysAdmin = 21;
  return (capEff >> kCapSysAdmin) & 1;
}

} // namespace parcagpu

#endif // PARCAGPU_PROFILING_ACCESS_H_
