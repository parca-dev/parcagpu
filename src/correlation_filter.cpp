#include "correlation_filter.h"

namespace parcagpu {

//=============================================================================
// CorrelationFilter implementation
//=============================================================================

void CorrelationFilter::insert(uint32_t correlation_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  set_.insert(correlation_id);
}

bool CorrelationFilter::check_and_remove(uint32_t correlation_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = set_.find(correlation_id);
  if (it != set_.end()) {
    set_.erase(it);
    return true;
  }
  return false;
}

size_t CorrelationFilter::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return set_.size();
}

//=============================================================================
// GraphCorrelationEntry implementation
//=============================================================================

GraphCorrelationEntry::GraphCorrelationEntry(uint32_t cycle)
    : state{GRAPH_STATE_UNINITIALIZED, GRAPH_STATE_UNINITIALIZED},
      ever_seen_kernel(false), insertion_cycle(cycle) {}

//=============================================================================
// GraphCorrelationMap implementation
//=============================================================================

GraphCorrelationMap::GraphCorrelationMap() : current_cycle_(0) {}

void GraphCorrelationMap::insert(uint32_t correlation_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  map_.emplace(correlation_id, GraphCorrelationEntry(current_cycle_));
}

void GraphCorrelationMap::cycle_start(uint32_t cycle) {
  std::lock_guard<std::mutex> lock(mutex_);
  current_cycle_ = cycle;
  uint32_t slot = cycle % 2;
  for (auto &pair : map_) {
    pair.second.state[slot] = GRAPH_STATE_CYCLE_CLEARED;
  }
}

bool GraphCorrelationMap::check_and_mark_seen(uint32_t correlation_id,
                                               uint32_t cycle) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = map_.find(correlation_id);
  if (it != map_.end()) {
    uint32_t slot = cycle % 2;
    it->second.state[slot] = GRAPH_STATE_KERNEL_SEEN;
    it->second.ever_seen_kernel = true;
    return true;
  }
  return false;
}

void GraphCorrelationMap::cycle_end() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = map_.begin(); it != map_.end();) {
    bool should_remove = false;
    if (it->second.state[0] == GRAPH_STATE_CYCLE_CLEARED &&
        it->second.state[1] == GRAPH_STATE_CYCLE_CLEARED) {
      if (it->second.ever_seen_kernel) {
        // Graph completed normally
        should_remove = true;
      } else if ((current_cycle_ - it->second.insertion_cycle) > 100) {
        // Fallback: never saw kernels and entry is very old
        should_remove = true;
      }
    }
    if (should_remove) {
      it = map_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t GraphCorrelationMap::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return map_.size();
}

} // namespace parcagpu
