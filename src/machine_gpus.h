// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_MACHINE_GPUS_H_
#define PARCAGPU_MACHINE_GPUS_H_

#include <cstdint>
#include <dirent.h>

namespace parcagpu {

// The NVIDIA GPUs on the machine: one directory per GPU (its PCI address)
// under /proc/driver/nvidia/gpus. Not what CUDA shows this process, which
// CUDA_VISIBLE_DEVICES often narrows to one GPU per worker process. 0 when
// the directory is missing (no driver, or a mock).
inline uint32_t countMachineGpus(const char *dir = "/proc/driver/nvidia/gpus") {
  DIR *d = opendir(dir);
  if (!d)
    return 0;
  uint32_t n = 0;
  while (struct dirent *e = readdir(d)) {
    if (e->d_name[0] != '.')
      n++;
  }
  closedir(d);
  return n;
}

// A default launch probe rate limit (kernel or graph launches), per thread:
// base split across the machine's GPUs. A machine usually has a launching
// thread per GPU (one process per GPU, or one thread per GPU in a single
// process), so the whole machine then sends about base events per second,
// however many GPUs it has.
inline double splitRateLimit(double base, uint32_t machineGpus) {
  return machineGpus > 1 ? base / machineGpus : base;
}

} // namespace parcagpu

#endif // PARCAGPU_MACHINE_GPUS_H_
