#include "correlation_filter.h"
#include <unordered_set>
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

} // extern "C"
