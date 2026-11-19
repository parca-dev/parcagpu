// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_DEVICE_INFO_H_
#define PARCAGPU_DEVICE_INFO_H_

#include <cuda.h>

// Maps this process's CUDA device ordinals to stable device identities (the
// UUID NVML reports as hw.id, and the PCI bus ID). Ordinals depend on
// CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER, so a profiler can only join
// per-process data (kernel timings, PC samples, NVTX spans) to host-wide GPU
// metrics through this mapping. Independent of PC sampling.
namespace parcagpu::device_info {

// Records the context's device and fires the device_info probe for it, once
// per device.
void onContextCreated(CUcontext ctx);

// Re-fires device_info for every known device when a consumer attaches (the
// probe's USDT semaphore count goes up), and periodically while one is
// attached. Cheap when nothing is attached; call it from API callbacks.
void emit();

} // namespace parcagpu::device_info

#endif // PARCAGPU_DEVICE_INFO_H_
