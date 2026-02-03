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

#ifdef __cplusplus
}
#endif
