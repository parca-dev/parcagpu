#ifndef PARCAGPU_STALL_REASON_MAP_H_
#define PARCAGPU_STALL_REASON_MAP_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace parcagpu {

// Contiguous, BPF-friendly stall reason name table.
// Array of fixed-width 64-byte name slots indexed directly by stall reason
// index. BPF reads names[stallReasonIndex * 64] — no pointer chasing.
static constexpr uint32_t STALL_REASON_NAME_LEN = 64;

class StallReasonMap {
public:
  StallReasonMap() = default;
  ~StallReasonMap() { std::free(buf); }

  StallReasonMap(const StallReasonMap &) = delete;
  StallReasonMap &operator=(const StallReasonMap &) = delete;

  // Build from parallel arrays (as returned by CUPTI).
  // Indices must be dense 0..N-1.
  void build(uint32_t numReasons, const uint32_t *indices, char **names) {
    // Find max index to size the array.
    uint32_t maxIdx = 0;
    for (uint32_t i = 0; i < numReasons; i++) {
      if (indices[i] > maxIdx)
        maxIdx = indices[i];
    }
    count = maxIdx + 1;

    std::free(buf);
    bufSize = count * STALL_REASON_NAME_LEN;
    buf = static_cast<char *>(std::calloc(1, bufSize));

    for (uint32_t i = 0; i < numReasons; i++) {
      char *slot = buf + indices[i] * STALL_REASON_NAME_LEN;
      strncpy(slot, names[i], STALL_REASON_NAME_LEN - 1);
    }
  }

  const char *data() const { return buf; }
  uint32_t size() const { return bufSize; }
  uint32_t numEntries() const { return count; }

private:
  char *buf = nullptr;
  uint32_t bufSize = 0;
  uint32_t count = 0;
};

} // namespace parcagpu

#endif // PARCAGPU_STALL_REASON_MAP_H_
