// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef AGENT_CTL_H_
#define AGENT_CTL_H_

#include <cstdint>

#include <cuda.h>

namespace parcagpu {

// Runtime control by parca-agent over procctl; the schema is in
// parcagpu_ctl.h. Every function is a no-op when the channel could not be set
// up, and parcagpu then runs on its environment variables alone.

// Creates the procctl region. Call once, at initialization.
void agentCtlInit();

// Picks up a new configuration from the agent and watches the agent's lease.
// Call at the top of every CUPTI callback: one atomic load when nothing
// changed.
void agentCtlPoll();

// Adds ctx's device to the device table reported to the agent. Call from
// CONTEXT_CREATED.
void agentCtlAddContext(CUcontext ctx);

// The agent's configuration. Inactive when the agent sent none or its lease
// expired; the environment decides then.
struct AgentConfig {
  bool active = false;
  uint32_t devices = 0; // parcagpu_ctl_config.pc_sampling_devices
  uint32_t factor = 0;
  uint32_t windowMs = 0;
};
AgentConfig agentCtlConfig();

} // namespace parcagpu

#endif // AGENT_CTL_H_
