#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to correlation filter
typedef void* CorrelationFilterHandle;

// Create a new correlation filter
CorrelationFilterHandle correlation_filter_create(void);

// Destroy the correlation filter
void correlation_filter_destroy(CorrelationFilterHandle filter);

// Insert a correlation ID into the filter
// Thread-safe: can be called from multiple threads concurrently
void correlation_filter_insert(CorrelationFilterHandle filter, uint32_t correlation_id);

// Check if a correlation ID exists and remove it if found
// Returns true if the correlation ID was found and removed, false otherwise
// Thread-safe: safe to call concurrently with inserts
bool correlation_filter_check_and_remove(CorrelationFilterHandle filter, uint32_t correlation_id);

// Get the current size of the filter (number of tracked correlation IDs)
// Note: This is an approximate count in concurrent scenarios
size_t correlation_filter_size(CorrelationFilterHandle filter);

// Graph correlation state values
enum GraphCorrelationState {
    GRAPH_STATE_UNINITIALIZED = 0,  // Entry just created, slot not yet processed
    GRAPH_STATE_CYCLE_CLEARED = 1,  // Cycle started, no kernels seen yet
    GRAPH_STATE_KERNEL_SEEN = 2     // At least one kernel seen this cycle
};

// Opaque handle to graph correlation map
typedef void* GraphCorrelationMapHandle;

// Create a new graph correlation map
GraphCorrelationMapHandle graph_correlation_map_create(void);

// Destroy the graph correlation map
void graph_correlation_map_destroy(GraphCorrelationMapHandle map);

// Insert a correlation ID into the map (called when sampling a graph launch)
// Thread-safe: can be called from multiple threads concurrently
void graph_correlation_map_insert(GraphCorrelationMapHandle map, uint32_t correlation_id);

// Start a new processing cycle - clears the appropriate slot for all entries
// Thread-safe
void graph_correlation_map_cycle_start(GraphCorrelationMapHandle map, uint32_t cycle);

// Check if correlation ID should fire probe and mark as seen for this cycle
// Returns true if the correlation ID is tracked (should fire probe)
// Thread-safe
bool graph_correlation_map_check_and_mark_seen(GraphCorrelationMapHandle map, uint32_t correlation_id, uint32_t cycle);

// End processing cycle - removes entries that haven't seen kernels in 2 consecutive cycles
// Thread-safe
void graph_correlation_map_cycle_end(GraphCorrelationMapHandle map);

// Get the current size of the map (number of tracked correlation IDs)
// Note: This is an approximate count in concurrent scenarios
size_t graph_correlation_map_size(GraphCorrelationMapHandle map);

// Get statistics about the map (for debugging)
// Returns the current size and age of the oldest entry (in cycles)
void graph_correlation_map_get_stats(GraphCorrelationMapHandle map, size_t* size, size_t* oldest_age);

#ifdef __cplusplus
}
#endif
