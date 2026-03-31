#include "probes.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "pc_sampling.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>

namespace parcagpu {

// CUDA driver version for 12.8.1 (when continuous PC sampling became stable)
// Version format: major * 1000 + minor * 10 + patch
#define CUDA_VERSION_12_8_1 12081

// Debug logging (reuse from main file)
extern bool debug_enabled;
extern void init_debug();

namespace {

// CUPTI helper functions (adapted from Proton's CuptiPCSamplingUtils.h)
// These wrap Proton's cupti API calls with PARCAGPU-specific setup

uint64_t getCubinCrc(const char *cubin, size_t size) {
  CUpti_GetCubinCrcParams cubinCrcParams = {
      /*size=*/CUpti_GetCubinCrcParamsSize,
      /*cubinSize=*/size,
      /*cubin=*/cubin,
      /*cubinCrc=*/0,
  };
  proton::cupti::getCubinCrc<true>(&cubinCrcParams);
  return cubinCrcParams.cubinCrc;
}

void enablePCSampling(CUcontext context) {
  CUpti_PCSamplingEnableParams params = {
      /*size=*/CUpti_PCSamplingEnableParamsSize,
      /*pPriv=*/NULL,
      /*ctx=*/context,
  };
  proton::cupti::pcSamplingEnable<true>(&params);
}

void startPCSampling(CUcontext context) {
  CUpti_PCSamplingStartParams params = {
      /*size=*/CUpti_PCSamplingStartParamsSize,
      /*pPriv=*/NULL,
      /*ctx=*/context,
  };
  proton::cupti::pcSamplingStart<true>(&params);
}

void stopPCSampling(CUcontext context) {
  CUpti_PCSamplingStopParams params = {
      /*size=*/CUpti_PCSamplingStopParamsSize,
      /*pPriv=*/NULL,
      /*ctx=*/context,
  };
  proton::cupti::pcSamplingStop<true>(&params);
}

void disablePCSampling(CUcontext context) {
  CUpti_PCSamplingDisableParams params = {
      /*size=*/CUpti_PCSamplingDisableParamsSize,
      /*pPriv=*/NULL,
      /*ctx=*/context,
  };
  proton::cupti::pcSamplingDisable<true>(&params);
}

// Returns true if data was retrieved successfully, false on error.
bool getPCSamplingData(CUcontext context,
                       CUpti_PCSamplingData *pcSamplingData) {
  CUpti_PCSamplingGetDataParams params = {
      /*size=*/CUpti_PCSamplingGetDataParamsSize,
      /*pPriv=*/NULL,
      /*ctx=*/context,
      /*pcSamplingData=*/pcSamplingData,
  };
  auto result = proton::cupti::pcSamplingGetData<false>(&params);
  if (result != CUPTI_SUCCESS) {
    DEBUG_PRINTF("cuptiPCSamplingGetData failed: error %d (ctx=%p)\n",
                 result, context);
    return false;
  }
  return true;
}

void setConfigurationAttribute(
    CUcontext context,
    std::vector<CUpti_PCSamplingConfigurationInfo> &configurationInfos) {
  CUpti_PCSamplingConfigurationInfoParams infoParams = {
      /*size=*/CUpti_PCSamplingConfigurationInfoParamsSize,
      /*pPriv=*/NULL,
      /*ctx=*/context,
      /*numAttributes=*/configurationInfos.size(),
      /*pPCSamplingConfigurationInfo=*/configurationInfos.data(),
  };
  proton::cupti::pcSamplingSetConfigurationAttribute<true>(&infoParams);
}

std::tuple<uint32_t, std::string, std::string>
getSassToSourceCorrelation(const char *functionName, uint64_t pcOffset,
                           const char *cubin, size_t cubinSize) {
  CUpti_GetSassToSourceCorrelationParams sassToSourceParams = {
      /*size=*/CUpti_GetSassToSourceCorrelationParamsSize,
      /*cubin=*/cubin,
      /*functionName=*/functionName,
      /*cubinSize=*/cubinSize,
      /*lineNumber=*/0,
      /*pcOffset=*/pcOffset,
      /*fileName=*/NULL,
      /*dirName=*/NULL,
  };
  // Get source can fail if the line mapping is not available
  proton::cupti::getSassToSourceCorrelation<false>(&sassToSourceParams);
  auto fileNameStr = sassToSourceParams.fileName
                         ? std::string(sassToSourceParams.fileName)
                         : "";
  auto dirNameStr =
      sassToSourceParams.dirName ? std::string(sassToSourceParams.dirName) : "";
  // Free the memory
  if (sassToSourceParams.fileName)
    std::free(sassToSourceParams.fileName);
  if (sassToSourceParams.dirName)
    std::free(sassToSourceParams.dirName);
  return std::make_tuple(sassToSourceParams.lineNumber, fileNameStr,
                         dirNameStr);
}

// Double-checked locking helper
template <typename CheckFn, typename ActionFn>
void doubleCheckedLock(CheckFn check, std::mutex &mutex, ActionFn action) {
  if (check()) {
    std::lock_guard<std::mutex> lock(mutex);
    if (check()) {
      action();
    }
  }
}

// Helper to get PARCAGPU's custom sampling frequency from environment
uint32_t getGPUSamplingFrequency() {
  // Default frequency for PARCAGPU is 18 (Proton uses 10)
  constexpr uint32_t PARCAGPU_DEFAULT_FREQUENCY = 18;

  uint32_t samplingPeriod = PARCAGPU_DEFAULT_FREQUENCY;
  const char *sampling_factor_env = getenv("PARCAGPU_SAMPLING_FACTOR");
  if (sampling_factor_env) {
    int factor = atoi(sampling_factor_env);
    if (factor >= 5 && factor <= 31) {
      samplingPeriod = factor;
      DEBUG_PRINTF("Using PARCAGPU_SAMPLING_FACTOR=%u\n", samplingPeriod);
    } else if (factor != 0) { // 0 is handled in isSupported()
      fprintf(stderr,
              "[PARCAGPU] Warning: PARCAGPU_SAMPLING_FACTOR=%d out of range "
              "[5,31], using default %u\n",
              factor, PARCAGPU_DEFAULT_FREQUENCY);
    }
  } else {
    return 0;
  }
  return samplingPeriod;
}

// Get number of stall reasons
size_t getNumStallReasons(CUcontext context) {
  size_t numStallReasons = 0;
  CUpti_PCSamplingGetNumStallReasonsParams numStallReasonsParams = {
      /*size=*/CUpti_PCSamplingGetNumStallReasonsParamsSize,
      /*pPriv=*/NULL,
      /*ctx=*/context,
      /*numStallReasons=*/&numStallReasons};
  proton::cupti::pcSamplingGetNumStallReasons<true>(&numStallReasonsParams);
  return numStallReasons;
}

// Get stall reason names and indices
std::pair<char **, uint32_t *>
getStallReasonNamesAndIndices(CUcontext context, size_t numStallReasons) {
  char **stallReasonNames =
      static_cast<char **>(std::calloc(numStallReasons, sizeof(char *)));
  for (size_t i = 0; i < numStallReasons; i++) {
    stallReasonNames[i] = static_cast<char *>(
        std::calloc(CUPTI_STALL_REASON_STRING_SIZE, sizeof(char)));
  }
  uint32_t *stallReasonIndices =
      static_cast<uint32_t *>(std::calloc(numStallReasons, sizeof(uint32_t)));
  CUpti_PCSamplingGetStallReasonsParams stallReasonsParams = {
      /*size=*/CUpti_PCSamplingGetStallReasonsParamsSize,
      /*pPriv=*/NULL,
      /*ctx=*/context,
      /*numStallReasons=*/numStallReasons,
      /*stallReasonIndex=*/stallReasonIndices,
      /*stallReasons=*/stallReasonNames,
  };
  proton::cupti::pcSamplingGetStallReasons<true>(&stallReasonsParams);
  return std::make_pair(stallReasonNames, stallReasonIndices);
}

// Match stall reasons to indices (PARCAGPU emits all stall reasons)
size_t matchStallReasonsToIndices(
    size_t numStallReasons, char **stallReasonNames,
    uint32_t *stallReasonIndices,
    std::map<size_t, size_t> &stallReasonIndexToMetricIndex,
    std::set<size_t> &notIssuedStallReasonIndices) {
  // PARCAGPU emits all stall reasons
  size_t numValidStalls = 0;
  for (size_t i = 0; i < numStallReasons; i++) {
    std::string cuptiStallName = std::string(stallReasonNames[i]);
    bool notIssued = cuptiStallName.find("not_issued") != std::string::npos ||
                     cuptiStallName.find("Not Issued") != std::string::npos;

    if (notIssued)
      notIssuedStallReasonIndices.insert(stallReasonIndices[i]);
    stallReasonIndexToMetricIndex[stallReasonIndices[i]] = i;
    numValidStalls++;
  }
  return numValidStalls;
}

// Allocate PC sampling data buffer
CUpti_PCSamplingData allocPCSamplingData(size_t collectNumPCs,
                                         size_t numValidStallReasons) {
  CUpti_PCSamplingData pcSamplingData{
      /*size=*/sizeof(CUpti_PCSamplingData),
      /*collectNumPcs=*/collectNumPCs,
      /*totalSamples=*/0,
      /*droppedSamples=*/0,
      /*totalNumPcs=*/0,
      /*remainingNumPcs=*/0,
      /*rangeId=*/0,
      /*pPcData=*/
      static_cast<CUpti_PCSamplingPCData *>(
          std::calloc(collectNumPCs, sizeof(CUpti_PCSamplingPCData)))};
  for (size_t i = 0; i < collectNumPCs; ++i) {
    pcSamplingData.pPcData[i].stallReason =
        static_cast<CUpti_PCSamplingStallReason *>(std::calloc(
            numValidStallReasons, sizeof(CUpti_PCSamplingStallReason)));
  }
  return pcSamplingData;
}

} // namespace

// ConfigureData implementation

CUpti_PCSamplingConfigurationInfo ConfigureData::configureStallReasons() {
  numStallReasons = getNumStallReasons(context);
  std::tie(this->stallReasonNames, this->stallReasonIndices) =
      getStallReasonNamesAndIndices(context, numStallReasons);
  numValidStallReasons = matchStallReasonsToIndices(
      numStallReasons, stallReasonNames, stallReasonIndices,
      stallReasonIndexToMetricIndex, notIssuedStallReasonIndices);

  CUpti_PCSamplingConfigurationInfo stallReasonInfo{};
  stallReasonInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
  stallReasonInfo.attributeData.stallReasonData.stallReasonCount =
      numValidStallReasons;
  stallReasonInfo.attributeData.stallReasonData.pStallReasonIndex =
      stallReasonIndices;
  return stallReasonInfo;
}

CUpti_PCSamplingConfigurationInfo ConfigureData::configureSamplingPeriod() {
  CUpti_PCSamplingConfigurationInfo samplingPeriodInfo{};
  samplingPeriodInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;

  // Use PARCAGPU's custom sampling frequency
  uint32_t frequency = getGPUSamplingFrequency();

  samplingPeriodInfo.attributeData.samplingPeriodData.samplingPeriod =
      frequency;
  return samplingPeriodInfo;
}

CUpti_PCSamplingConfigurationInfo ConfigureData::configureSamplingBuffer() {
  CUpti_PCSamplingConfigurationInfo samplingBufferInfo{};
  samplingBufferInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  this->pcSamplingData =
      allocPCSamplingData(DataBufferPCCount, numValidStallReasons);
  samplingBufferInfo.attributeData.samplingDataBufferData.samplingDataBuffer =
      &this->pcSamplingData;
  return samplingBufferInfo;
}

CUpti_PCSamplingConfigurationInfo ConfigureData::configureScratchBuffer() {
  CUpti_PCSamplingConfigurationInfo scratchBufferInfo{};
  scratchBufferInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
  scratchBufferInfo.attributeData.scratchBufferSizeData.scratchBufferSize =
      ScratchBufferSize;
  return scratchBufferInfo;
}

CUpti_PCSamplingConfigurationInfo ConfigureData::configureHardwareBufferSize() {
  CUpti_PCSamplingConfigurationInfo hardwareBufferInfo{};
  hardwareBufferInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
  hardwareBufferInfo.attributeData.hardwareBufferSizeData.hardwareBufferSize =
      HardwareBufferSize;
  return hardwareBufferInfo;
}

CUpti_PCSamplingConfigurationInfo ConfigureData::configureCollectionMode() {
  CUpti_PCSamplingConfigurationInfo collectionModeInfo{};
  collectionModeInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
  collectionModeInfo.attributeData.collectionModeData.collectionMode =
      CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
  return collectionModeInfo;
}

void ConfigureData::initialize(CUcontext context) {
  this->context = context;
  proton::cupti::getContextId<true>(context, &contextId);

  DEBUG_PRINTF("Initializing PC sampling for context %p (id %u)\n", context,
               contextId);

  configurationInfos.emplace_back(configureStallReasons());
  configurationInfos.emplace_back(configureCollectionMode());
  configurationInfos.emplace_back(configureSamplingBuffer());
  // Don't set sampling period — let CUPTI use its default.
  // Explicit period values silently break sampling on some GPUs (e.g. Blackwell).

  setConfigurationAttribute(context, configurationInfos);

  // Allocate a separate output buffer for getPCSamplingData calls.
  // The configured pcSamplingData buffer is owned by CUPTI internally;
  // we must pass a different buffer to getPCSamplingData.
  this->outputData =
      allocPCSamplingData(DataBufferPCCount, numValidStallReasons);

  DEBUG_PRINTF("PC sampling configured with %u stall reasons (%u valid)\n",
               numStallReasons, numValidStallReasons);
}

// GPUPCSampling implementation

bool PCSampling::isSupported() {
  // PARCAGPU_SAMPLING_FACTOR must be set to enable PC sampling.
  // Any non-zero value enables it; 0 disables.
  // If unset, PC sampling is disabled.
  const char *env = getenv("PARCAGPU_SAMPLING_FACTOR");
  if (!env) {
    DEBUG_PRINTF("PC sampling not enabled (PARCAGPU_SAMPLING_FACTOR not set)\n");
    return false;
  }
  int factor = atoi(env);
  if (factor == 0) {
    DEBUG_PRINTF("PC sampling disabled via PARCAGPU_SAMPLING_FACTOR=0\n");
    return false;
  }

  // Check CUDA driver version >= 12.8.1
  int driverVersion = 0;
  proton::cuda::driverGetVersion<true>(&driverVersion);

  if (driverVersion < CUDA_VERSION_12_8_1) {
    int major = driverVersion / 1000;
    int minor = (driverVersion % 1000) / 10;
    int patch = driverVersion % 10;
    DEBUG_PRINTF("PC sampling not supported: CUDA driver version %d.%d.%d < "
                 "required 12.8.1\n",
                 major, minor, patch);
    return false;
  }

  int major = driverVersion / 1000;
  int minor = (driverVersion % 1000) / 10;
  int patch = driverVersion % 10;
  DEBUG_PRINTF("PC sampling supported: CUDA driver version %d.%d.%d\n", major,
               minor, patch);
  return true;
}

ConfigureData *PCSampling::getConfigureData(uint32_t contextId) {
  return &contextIdToConfigureData[contextId];
}

CubinData *PCSampling::getCubinData(uint64_t cubinCrc) {
  return &(cubinCrcToCubinData[cubinCrc].first);
}

void PCSampling::initialize(CUcontext context) {
  uint32_t contextId = 0;
  proton::cupti::getContextId<true>(context, &contextId);

  doubleCheckedLock(
      [&]() { return !contextInitialized.contain(contextId); }, contextMutex,
      [&]() {
        enablePCSampling(context);
        auto *configData = getConfigureData(contextId);
        configData->initialize(context);

        // Build contiguous stall reason map for USDT probe emission.
        stallReasonMap.build(configData->numStallReasons,
                            configData->stallReasonIndices,
                            configData->stallReasonNames);

        contextInitialized.insert(contextId);
        initializedContextIds.push_back(contextId);
        DEBUG_PRINTF("PC sampling started in continuous mode for context %u\n",
                     contextId);
      });
}

void PCSampling::processPCSamplingData(ConfigureData *configureData) {
  auto *pcSamplingData = &configureData->outputData;

  if (pcSamplingData->totalNumPcs == 0) {
    return;
  }

  DEBUG_PRINTF("Processing %zu PCs (remaining: %zu)\n",
               pcSamplingData->totalNumPcs, pcSamplingData->remainingNumPcs);

  // Process each PC sample
  for (size_t i = 0; i < pcSamplingData->totalNumPcs; ++i) {
    auto *pcData = pcSamplingData->pPcData + i;

    // Calculate total and stalled samples
    uint64_t totalSamples = 0;
    uint64_t stalledSamples = 0;

    for (size_t j = 0; j < pcData->stallReasonCount; ++j) {
      auto *stallReason = &pcData->stallReason[j];
      totalSamples += stallReason->samples;

      // Check if this is a "not_issued" stall (not really stalled)
      bool isNotIssued = configureData->notIssuedStallReasonIndices.count(
                             stallReason->pcSamplingStallReasonIndex) > 0;

      if (!isNotIssued) {
        stalledSamples += stallReason->samples;
      }
    }

    // Source correlation only for debug logging — backend resolves
    // file/line from the cubin using pcOffset.
    if (debug_enabled) {
      auto *cubinData = getCubinData(pcData->cubinCrc);
      auto key =
          CubinData::LineInfoKey{pcData->functionIndex, pcData->pcOffset};
      if (cubinData->lineInfo.find(key) == cubinData->lineInfo.end()) {
        auto [lineNumber, fileName, dirName] =
            getSassToSourceCorrelation(pcData->functionName, pcData->pcOffset,
                                       cubinData->cubin, cubinData->cubinSize);
        cubinData->lineInfo.try_emplace(key, lineNumber,
                                        std::string(pcData->functionName),
                                        dirName, fileName);
      }
      auto &lineInfo = cubinData->lineInfo[key];
      std::string fullPath = lineInfo.fileName.size()
                                 ? lineInfo.dirName + "/" + lineInfo.fileName
                                 : "";
      DEBUG_PRINTF("  [%zu] func=%s pc=0x%lx total=%lu stalled=%lu %s:%u\n",
                   i, lineInfo.functionName.c_str(), pcData->pcOffset,
                   totalSamples, stalledSamples, fullPath.c_str(),
                   lineInfo.lineNumber);
    }

    PARCAGPU_PC_SAMPLE_SUMMARY(pcData->functionIndex, pcData->pcOffset,
                               totalSamples, stalledSamples,
                               pcData->functionName);

    // Emit detailed stall reason probes
    for (size_t j = 0; j < pcData->stallReasonCount; ++j) {
      auto *stallReason = &pcData->stallReason[j];
      auto stallReasonIndex = stallReason->pcSamplingStallReasonIndex;

      if (debug_enabled) {
        const char *stallReasonName = "";
        for (size_t k = 0; k < configureData->numStallReasons; k++) {
          if (configureData->stallReasonIndices[k] == stallReasonIndex) {
            stallReasonName = configureData->stallReasonNames[k];
            break;
          }
        }
        if (stallReason->samples > 0) {
          DEBUG_PRINTF("    stall: %s = %u\n", stallReasonName,
                       stallReason->samples);
        }
      }

      PARCAGPU_PC_STALL_REASON(pcData->functionIndex, pcData->pcOffset,
                               stallReasonIndex, stallReason->samples);
    }
  }
}

void PCSampling::collectData(CUcontext context) {
  uint32_t contextId = 0;
  proton::cupti::getContextId<true>(context, &contextId);

  if (!contextInitialized.contain(contextId)) {
    DEBUG_PRINTF("Context %u not initialized, skipping data collection\n",
                 contextId);
    return;
  }

  auto *configureData = getConfigureData(contextId);
  DEBUG_PRINTF("Collecting PC sampling data for context %u (cfg total=%zu remaining=%zu)\n",
               contextId, configureData->pcSamplingData.totalNumPcs,
               configureData->pcSamplingData.remainingNumPcs);

  // Re-emit stall reason map periodically so the profiler backend can
  // join stall reason indices to human-readable names.
  if (stallReasonMap.data() && stallReasonMapLimiter.tryAcquire()) {
    PARCAGPU_STALL_REASON_MAP(stallReasonMap.data(),
                              stallReasonMap.numEntries());
  }

  // Use the separate output buffer for getData — the configured
  // pcSamplingData buffer is owned by CUPTI.
  bool ok = getPCSamplingData(context, &configureData->outputData);
  DEBUG_PRINTF("getData: ok=%d output total=%zu remaining=%zu "
               "cfg total=%zu remaining=%zu\n",
               ok, configureData->outputData.totalNumPcs,
               configureData->outputData.remainingNumPcs,
               configureData->pcSamplingData.totalNumPcs,
               configureData->pcSamplingData.remainingNumPcs);
  processPCSamplingData(configureData);
}

void PCSampling::collectAllData() {
  std::lock_guard<std::mutex> lock(contextMutex);
  for (auto contextId : initializedContextIds) {
    auto result = contextIdToConfigureData.find(contextId);
    if (!result) {
      DEBUG_PRINTF("Context %u in initializedContextIds but not in map, "
                   "skipping\n", contextId);
      continue;
    }
    auto *configureData = &result->get();
    DEBUG_PRINTF("Draining PC sampling data for context %u\n", contextId);
    processPCSamplingData(configureData);
  }
}

void PCSampling::finalize(CUcontext context) {
  uint32_t contextId = 0;
  proton::cupti::getContextId<true>(context, &contextId);

  if (!contextInitialized.contain(contextId))
    return;

  // Hold contextMutex for the entire finalize to prevent collectAllData
  // from racing with us (it iterates initializedContextIds under this lock).
  std::lock_guard<std::mutex> lock(contextMutex);

  DEBUG_PRINTF("Finalizing PC sampling for context %p\n", context);

  // Remove from iteration list first so collectAllData won't touch this context
  initializedContextIds.erase(
      std::remove(initializedContextIds.begin(),
                  initializedContextIds.end(), contextId),
      initializedContextIds.end());

  // Drain remaining data before disabling
  auto *configureData = getConfigureData(contextId);
  processPCSamplingData(configureData);

  // After disable, CUPTI may fill remaining records — drain once more
  if (configureData->pcSamplingData.totalNumPcs > 0) {
    processPCSamplingData(configureData);
  }

  contextIdToConfigureData.erase(contextId);
  contextInitialized.erase(contextId);
}

void PCSampling::loadModule(const char *cubin, size_t cubinSize) {
  auto cubinCrc = getCubinCrc(cubin, cubinSize);
  auto *cubinData = getCubinData(cubinCrc);

  if (cubinCrcToCubinData.contain(cubinCrc)) {
    // Increment reference count
    cubinCrcToCubinData[cubinCrc].second++;
    DEBUG_PRINTF("Module 0x%lx loaded (refcount=%zu)\n", cubinCrc,
                 cubinCrcToCubinData[cubinCrc].second);
  } else {
    // New module
    cubinData->cubinCrc = cubinCrc;
    cubinData->cubinSize = cubinSize;
    cubinData->cubin = cubin;
    cubinCrcToCubinData[cubinCrc].second = 1;
    DEBUG_PRINTF("Module 0x%lx loaded (new)\n", cubinCrc);
    PARCAGPU_CUBIN_LOADED(cubinCrc, 0, 0);
  }
}

void PCSampling::unloadModule(const char *cubin, size_t cubinSize) {
  auto cubinCrc = getCubinCrc(cubin, cubinSize);

  if (!cubinCrcToCubinData.contain(cubinCrc))
    return;

  auto count = cubinCrcToCubinData[cubinCrc].second;
  if (count > 1) {
    cubinCrcToCubinData[cubinCrc].second = count - 1;
    DEBUG_PRINTF("Module 0x%lx unloaded (refcount=%zu)\n", cubinCrc, count - 1);
  } else {
    cubinCrcToCubinData.erase(cubinCrc);
    DEBUG_PRINTF("Module 0x%lx unloaded (removed)\n", cubinCrc);
    PARCAGPU_CUBIN_UNLOADED(cubinCrc, 0);
  }
}

} // namespace parcagpu
