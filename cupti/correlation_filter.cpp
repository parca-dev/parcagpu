#include "correlation_filter.h"
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <memory>

// CorrelationFilter implementation using std::unordered_set with mutex protection
// This provides thread-safe access with minimal overhead for our use case
class CorrelationFilter {
public:
    CorrelationFilter() = default;

    // Insert a correlation ID into the filter
    // Thread-safe
    void insert(uint32_t correlation_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        set_.insert(correlation_id);
    }

    // Check if correlation ID exists and remove it atomically
    // Returns true if found and removed, false if not found
    // Thread-safe
    bool check_and_remove(uint32_t correlation_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = set_.find(correlation_id);
        if (it != set_.end()) {
            set_.erase(it);
            return true;
        }
        return false;
    }

    // Get current size
    // Thread-safe
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return set_.size();
    }

private:
    std::unordered_set<uint32_t> set_;
    mutable std::mutex mutex_;
};

// GraphCorrelationMap implementation for tracking graph launches across buffer cycles
// Uses a 2-slot state machine per correlation ID to detect when graph launches are complete
struct GraphCorrelationEntry {
    uint8_t state[2];         // State for alternating cycles
    bool ever_seen_kernel;    // True once we've seen at least one kernel activity
    uint32_t insertion_cycle; // Buffer cycle when entry was created (for fallback cleanup)

    GraphCorrelationEntry(uint32_t cycle)
        : state{GRAPH_STATE_UNINITIALIZED, GRAPH_STATE_UNINITIALIZED}
        , ever_seen_kernel(false)
        , insertion_cycle(cycle) {}
};

class GraphCorrelationMap {
public:
    GraphCorrelationMap() : current_cycle_(0) {}

    // Insert a correlation ID (called when sampling a graph launch)
    // Thread-safe
    void insert(uint32_t correlation_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        map_.emplace(correlation_id, GraphCorrelationEntry(current_cycle_));
    }

    // Start a new processing cycle - clear the appropriate slot for all entries
    // Thread-safe
    void cycle_start(uint32_t cycle) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_cycle_ = cycle;
        uint32_t slot = cycle % 2;
        for (auto& pair : map_) {
            pair.second.state[slot] = GRAPH_STATE_CYCLE_CLEARED;
        }
    }

    // Check if correlation ID is tracked and mark as seen for this cycle
    // Returns true if tracked (should fire probe)
    // Thread-safe
    bool check_and_mark_seen(uint32_t correlation_id, uint32_t cycle) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(correlation_id);
        if (it != map_.end()) {
            uint32_t slot = cycle % 2;
            it->second.state[slot] = GRAPH_STATE_KERNEL_SEEN;
            it->second.ever_seen_kernel = true;  // Mark that we've seen at least one kernel
            return true;
        }
        return false;
    }

    // End processing cycle - remove entries based on two conditions:
    // 1. Primary: Both slots CYCLE_CLEARED AND we've seen at least one kernel (graph completed)
    // 2. Fallback: Both slots CYCLE_CLEARED AND never seen kernel AND age > 100 cycles
    //    (handles GPU reset, failed launches, etc.)
    // Thread-safe
    void cycle_end() {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t removed_normal = 0;
        size_t removed_fallback = 0;

        for (auto it = map_.begin(); it != map_.end(); ) {
            bool should_remove = false;
            bool is_fallback = false;

            if (it->second.state[0] == GRAPH_STATE_CYCLE_CLEARED &&
                it->second.state[1] == GRAPH_STATE_CYCLE_CLEARED) {

                if (it->second.ever_seen_kernel) {
                    // Primary: Graph completed normally (saw kernels, then stopped)
                    should_remove = true;
                    removed_normal++;
                } else if ((current_cycle_ - it->second.insertion_cycle) > 100) {
                    // Fallback: Never saw kernels and entry is very old (>100 cycles)
                    // Prevents leaking entries when GPU resets or launches fail
                    should_remove = true;
                    is_fallback = true;
                    removed_fallback++;
                }
            }

            if (should_remove) {
                it = map_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Get cleanup stats (for debugging)
    void get_stats(size_t& size, size_t& oldest_age) const {
        std::lock_guard<std::mutex> lock(mutex_);
        size = map_.size();
        oldest_age = 0;
        for (const auto& pair : map_) {
            uint32_t age = current_cycle_ - pair.second.insertion_cycle;
            if (age > oldest_age) {
                oldest_age = age;
            }
        }
    }

    // Get current size
    // Thread-safe
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_.size();
    }

private:
    std::unordered_map<uint32_t, GraphCorrelationEntry> map_;
    uint32_t current_cycle_;
    mutable std::mutex mutex_;
};

// C API implementation
extern "C" {

CorrelationFilterHandle correlation_filter_create(void) {
    return new CorrelationFilter();
}

void correlation_filter_destroy(CorrelationFilterHandle filter) {
    if (filter) {
        delete static_cast<CorrelationFilter*>(filter);
    }
}

void correlation_filter_insert(CorrelationFilterHandle filter, uint32_t correlation_id) {
    if (filter) {
        static_cast<CorrelationFilter*>(filter)->insert(correlation_id);
    }
}

bool correlation_filter_check_and_remove(CorrelationFilterHandle filter, uint32_t correlation_id) {
    if (filter) {
        return static_cast<CorrelationFilter*>(filter)->check_and_remove(correlation_id);
    }
    return false;
}

size_t correlation_filter_size(CorrelationFilterHandle filter) {
    if (filter) {
        return static_cast<CorrelationFilter*>(filter)->size();
    }
    return 0;
}

GraphCorrelationMapHandle graph_correlation_map_create(void) {
    return new GraphCorrelationMap();
}

void graph_correlation_map_destroy(GraphCorrelationMapHandle map) {
    if (map) {
        delete static_cast<GraphCorrelationMap*>(map);
    }
}

void graph_correlation_map_insert(GraphCorrelationMapHandle map, uint32_t correlation_id) {
    if (map) {
        static_cast<GraphCorrelationMap*>(map)->insert(correlation_id);
    }
}

void graph_correlation_map_cycle_start(GraphCorrelationMapHandle map, uint32_t cycle) {
    if (map) {
        static_cast<GraphCorrelationMap*>(map)->cycle_start(cycle);
    }
}

bool graph_correlation_map_check_and_mark_seen(GraphCorrelationMapHandle map, uint32_t correlation_id, uint32_t cycle) {
    if (map) {
        return static_cast<GraphCorrelationMap*>(map)->check_and_mark_seen(correlation_id, cycle);
    }
    return false;
}

void graph_correlation_map_cycle_end(GraphCorrelationMapHandle map) {
    if (map) {
        static_cast<GraphCorrelationMap*>(map)->cycle_end();
    }
}

size_t graph_correlation_map_size(GraphCorrelationMapHandle map) {
    if (map) {
        return static_cast<GraphCorrelationMap*>(map)->size();
    }
    return 0;
}

void graph_correlation_map_get_stats(GraphCorrelationMapHandle map, size_t* size, size_t* oldest_age) {
    if (map && size && oldest_age) {
        static_cast<GraphCorrelationMap*>(map)->get_stats(*size, *oldest_age);
    }
}

} // extern "C"
