#ifndef PARCAGPU_PC_SAMPLING_H_
#define PARCAGPU_PC_SAMPLING_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <sys/sdt.h>
#include <vector>

#include <cuda.h>
#include <cupti.h>

#include "Driver/GPU/CuptiApi.h"
#include "Profiler/Cupti/CuptiPCSampling.h"
#include "Utility/Map.h"
#include "Utility/Set.h"
#include "Utility/Singleton.h"

namespace parcagpu {

// Use Proton's CubinData directly
using proton::CubinData;

// ConfigureData for PARCAGPU (based on Proton's but standalone)
// We don't inherit to avoid linking Proton's profiler dependencies
struct ConfigureData {
  ConfigureData() = default;

  ~ConfigureData() {
    if (stallReasonNames) {
      for (size_t i = 0; i < numStallReasons; i++) {
        if (stallReasonNames[i])
          std::free(stallReasonNames[i]);
      }
      std::free(stallReasonNames);
    }
    if (stallReasonIndices)
      std::free(stallReasonIndices);
    if (pcSamplingData.pPcData) {
      for (size_t i = 0; i < numValidStallReasons; ++i) {
        std::free(pcSamplingData.pPcData[i].stallReason);
      }
      std::free(pcSamplingData.pPcData);
    }
  }

  void initialize(CUcontext context);

  CUpti_PCSamplingConfigurationInfo configureStallReasons();
  CUpti_PCSamplingConfigurationInfo configureSamplingPeriod();
  CUpti_PCSamplingConfigurationInfo configureSamplingBuffer();
  CUpti_PCSamplingConfigurationInfo configureScratchBuffer();
  CUpti_PCSamplingConfigurationInfo configureHardwareBufferSize();
  CUpti_PCSamplingConfigurationInfo configureCollectionMode();

  // Buffer size constants (from Proton)
  static constexpr size_t HardwareBufferSize = 128 * 1024 * 1024;
  static constexpr size_t ScratchBufferSize = 16 * 1024 * 1024;
  static constexpr size_t DataBufferPCCount = 1024;

  CUcontext context{};
  uint32_t contextId;
  uint32_t numStallReasons{};
  uint32_t numValidStallReasons{};
  char **stallReasonNames{};
  uint32_t *stallReasonIndices{};
  std::map<size_t, size_t> stallReasonIndexToMetricIndex{};
  std::set<size_t> notIssuedStallReasonIndices{};
  CUpti_PCSamplingData pcSamplingData{};
  std::vector<CUpti_PCSamplingConfigurationInfo> configurationInfos;
};

// PC Sampling singleton class (adapted from Proton's CuptiPCSampling)
class PCSampling : public proton::Singleton<PCSampling> {
public:
  PCSampling() = default;
  virtual ~PCSampling() = default;

  // Check if PC sampling is supported (CUPTI >= 12.8.1)
  static bool isSupported();

  void initialize(CUcontext context);
  void collectData(CUcontext context, uint32_t correlationId);
  void finalize(CUcontext context);
  void loadModule(const char *cubin, size_t cubinSize);
  void unloadModule(const char *cubin, size_t cubinSize);

private:
  ConfigureData *getConfigureData(uint32_t contextId);
  CubinData *getCubinData(uint64_t cubinCrc);
  void processPCSamplingData(ConfigureData *configureData,
                             uint32_t correlationId);

  proton::ThreadSafeMap<uint32_t, ConfigureData> contextIdToConfigureData;
  proton::ThreadSafeMap<size_t, std::pair<CubinData, size_t>>
      cubinCrcToCubinData;
  proton::ThreadSafeSet<uint32_t> contextInitialized;

  std::mutex contextMutex{};
};

} // namespace parcagpu

#endif // PARCAGPU_PC_SAMPLING_H_
