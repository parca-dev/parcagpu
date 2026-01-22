#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace parcagpu {

//=============================================================================
// Correlation Filter - tracks which kernel launches we've sampled
//=============================================================================

// CorrelationFilter for non-graph kernel launches
// Insert on API callback, check-and-remove on kernel activity
class CorrelationFilter {
public:
  void insert(uint32_t correlation_id);
  bool check_and_remove(uint32_t correlation_id);
  size_t size() const;

private:
  std::unordered_set<uint32_t> set_;
  mutable std::mutex mutex_;
};

// Graph correlation state values
enum GraphCorrelationState {
  GRAPH_STATE_UNINITIALIZED = 0, // Entry just created, slot not yet processed
  GRAPH_STATE_CYCLE_CLEARED = 1, // Cycle started, no kernels seen yet
  GRAPH_STATE_KERNEL_SEEN = 2    // At least one kernel seen this cycle
};

// GraphCorrelationMap for graph launches (multiple kernels per launch)
// Uses 2-slot state machine to detect when all kernels from a graph have arrived
struct GraphCorrelationEntry {
  uint8_t state[2];         // State for alternating cycles
  bool ever_seen_kernel;    // True once we've seen at least one kernel activity
  uint32_t insertion_cycle; // Buffer cycle when entry was created

  GraphCorrelationEntry(uint32_t cycle);
};

class GraphCorrelationMap {
public:
  GraphCorrelationMap();

  void insert(uint32_t correlation_id);
  void cycle_start(uint32_t cycle);
  bool check_and_mark_seen(uint32_t correlation_id, uint32_t cycle);
  void cycle_end();
  size_t size() const;

private:
  std::unordered_map<uint32_t, GraphCorrelationEntry> map_;
  uint32_t current_cycle_;
  mutable std::mutex mutex_;
};

} // namespace parcagpu
