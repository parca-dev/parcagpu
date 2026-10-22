// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef AGENT_CTL_H_
#define AGENT_CTL_H_

#include <cstdint>
#include <functional>

#include <cuda.h>

namespace parcagpu {

// Runtime control by parca-agent over procctl; the schema is in
// parcagpu_ctl.h. Every function is a no-op when the channel could not be set
// up, and parcagpu then runs on its environment variables alone.

// Creates the procctl region. Call once, early in initialization.
void agentCtlInit();

// True when the agent channel is up.
bool agentCtlReady();

// Applies a configuration change: must call commit(), which makes the new
// configuration current, and stop whatever it no longer allows, before it
// returns; the change is acked after. Called from agentCtlPoll.
using AgentApplyFn = void (*)(const std::function<void()> &commit);

// Sets the apply function and starts the release thread: every 100 ms it does
// what a CUDA callback would, so a change (above all, giving up a GPU) takes
// effect even while the application makes no CUDA calls.
void agentCtlStart(AgentApplyFn apply);

// Stops the release thread. Call before tearing down.
void agentCtlShutdown();

// Picks up a new configuration from the agent and watches the agent's lease.
// Call at the top of every CUPTI callback: one atomic load when nothing
// changed.
void agentCtlPoll();

// Adds ctx's device to the device table reported to the agent. Call from
// CONTEXT_CREATED.
void agentCtlAddContext(CUcontext ctx);

// The agent's configuration. Inactive when the agent sent none or its lease
// expired; the environment decides then. gen counts configuration changes.
struct AgentConfig {
  bool active = false;
  uint32_t devices = 0; // parcagpu_ctl_config.pc_sampling_devices
  uint32_t factor = 0;
  uint32_t windowMs = 0;
  uint32_t gapMs = 0;
};
AgentConfig agentCtlConfig();
uint64_t agentCtlGeneration();

// Whether ctx may PC sample: the agent's configuration if active, else
// envDefault.
bool agentCtlAllows(CUcontext ctx, bool envDefault);

// Reported state, per device: pc_capable, and for ctx's device its PC
// sampling state (PARCAGPU_PC_*) and applied sampling factor.
void agentCtlSetCapable(bool capable);
void agentCtlSetPCState(CUcontext ctx, uint32_t state);
void agentCtlSetFactor(CUcontext ctx, uint32_t factor);

// Makes ctx current on this thread for CUPTI calls on it, and undoes that.
bool agentCtlPushContext(CUcontext ctx);
void agentCtlPopContext();

} // namespace parcagpu

#endif // AGENT_CTL_H_
