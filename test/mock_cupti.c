#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cupti.h>
#include <cupti_pcsampling.h>

// Define callback function types if not already defined by CUPTI headers
#ifndef CUpti_BufferRequestFunc
typedef void (*CUpti_BufferRequestFunc)(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
#endif

#ifndef CUpti_BufferCompletedFunc
typedef void (*CUpti_BufferCompletedFunc)(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
#endif

// Global storage for registered callbacks (exported for test access)
CUpti_CallbackFunc __cupti_runtime_api_callback = NULL;
void *__cupti_runtime_api_userdata = NULL;
CUpti_BufferRequestFunc __cupti_buffer_requested_callback = NULL;
CUpti_BufferCompletedFunc __cupti_buffer_completed_callback = NULL;

// Mock implementations of CUPTI APIs used by cupti-prof.c

CUptiResult cuptiActivityFlushPeriod(uint32_t period) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiActivityFlushPeriod(%u)\n", period);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiGetResultString(CUptiResult result, const char **str) {
    static const char *success = "CUPTI_SUCCESS";
    static const char *error = "CUPTI_ERROR";
    *str = (result == CUPTI_SUCCESS) ? success : error;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                           CUpti_CallbackFunc callback,
                           void *userdata) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiSubscribe()\n");
    __cupti_runtime_api_callback = callback;
    __cupti_runtime_api_userdata = userdata;
    *subscriber = (CUpti_SubscriberHandle)0x1234;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid) {
    (void)subscriber;  // Mark as intentionally unused
    fprintf(stderr, "[MOCK_CUPTI] cuptiEnableCallback(enable=%u, domain=%u, cbid=%u)\n",
            enable, domain, cbid);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BufferRequestFunc funcBufferRequested,
    CUpti_BufferCompletedFunc funcBufferCompleted) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiActivityRegisterCallbacks()\n");
    __cupti_buffer_requested_callback = funcBufferRequested;
    __cupti_buffer_completed_callback = funcBufferCompleted;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiActivityEnable(kind=%u)\n", kind);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityFlushAll(uint32_t flag) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiActivityFlushAll(flag=%u)\n", flag);
    // Don't actually call any callbacks during flush to avoid issues during cleanup
    (void)flag;
    return CUPTI_SUCCESS;
}

// Track iteration state per buffer
typedef struct {
    uint8_t *buffer;
    size_t offset;
} BufferIterState;

static BufferIterState iter_state = {NULL, 0};

CUptiResult cuptiActivityGetNextRecord(uint8_t *buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity **record) {
    // Reset state if this is a new buffer
    if (iter_state.buffer != buffer) {
        iter_state.buffer = buffer;
        iter_state.offset = 0;
    }

    // Check if we've reached the end
    if (iter_state.offset >= validBufferSizeBytes) {
        iter_state.buffer = NULL;
        iter_state.offset = 0;
        return CUPTI_ERROR_MAX_LIMIT_REACHED;
    }

    // Get the record at current offset
    CUpti_Activity *activity = (CUpti_Activity *)(buffer + iter_state.offset);

    // Determine record size based on kind
    size_t recordSize = 0;
    switch (activity->kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        case CUPTI_ACTIVITY_KIND_KERNEL:
            // Use CUpti_ActivityKernel5 which includes graphId and graphNodeId
            recordSize = sizeof(CUpti_ActivityKernel5);
            break;
        case CUPTI_ACTIVITY_KIND_GRAPH_TRACE:
            recordSize = sizeof(CUpti_ActivityGraphTrace);
            break;
        default:
            // Unknown kind, can't continue
            iter_state.buffer = NULL;
            iter_state.offset = 0;
            return CUPTI_ERROR_INVALID_KIND;
    }

    // Advance offset for next call
    iter_state.offset += recordSize;

    *record = activity;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext context,
                                              uint32_t streamId,
                                              size_t *dropped) {
    (void)context;  // Mark as intentionally unused
    (void)streamId;
    *dropped = 0;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiGetContextId(CUcontext context, uint32_t *contextId) {
    (void)context;  // Mark as intentionally unused
    // Return a fixed context ID for testing
    *contextId = 1;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
    (void)subscriber;  // Mark as intentionally unused
    fprintf(stderr, "[MOCK_CUPTI] cuptiUnsubscribe()\n");
    return CUPTI_SUCCESS;
}

// =========================================================================
// PC Sampling mock — uses a real cubin from pc_sample_toy for realistic
// CRC, offsets, and source-line correlation.
// =========================================================================

// Real cubin loaded from file (set MOCK_CUBIN_PATH, or auto-detected).
static char *__cubin_data = NULL;
static size_t __cubin_size = 0;
static uint64_t __cubin_crc = 0;

// CRC function shared between cuptiGetCubinCrc and internal use.
static uint64_t __compute_crc(const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t crc = 0xDEADBEEFULL;
    for (size_t i = 0; i < size; i++)
        crc = crc * 31 + bytes[i];
    return crc;
}

static void __load_cubin(void) {
    if (__cubin_data)
        return;
    const char *path = getenv("MOCK_CUBIN_PATH");
    if (!path) {
        fprintf(stderr, "[MOCK_CUPTI] MOCK_CUBIN_PATH not set, no cubin loaded\n");
        return;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[MOCK_CUPTI] Failed to open cubin: %s\n", path);
        return;
    }
    fseek(f, 0, SEEK_END);
    __cubin_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    __cubin_data = (char *)malloc(__cubin_size);
    if (fread(__cubin_data, 1, __cubin_size, f) != __cubin_size) {
        fprintf(stderr, "[MOCK_CUPTI] Short read on cubin: %s\n", path);
        free(__cubin_data);
        __cubin_data = NULL;
        __cubin_size = 0;
        fclose(f);
        return;
    }
    fclose(f);
    __cubin_crc = __compute_crc(__cubin_data, __cubin_size);
    fprintf(stderr, "[MOCK_CUPTI] Loaded cubin: %s (%zu bytes, crc=0x%lx)\n",
            path, __cubin_size, __cubin_crc);
}

CUptiResult cuptiGetVersion(uint32_t *version) {
    *version = 24;
    return CUPTI_SUCCESS;
}

static int __pc_sampling_started = 0;

CUptiResult cuptiPCSamplingEnable(CUpti_PCSamplingEnableParams *params) {
    (void)params;
    fprintf(stderr, "[MOCK_CUPTI] cuptiPCSamplingEnable()\n");
    return CUPTI_SUCCESS;
}

CUptiResult cuptiPCSamplingDisable(CUpti_PCSamplingDisableParams *params) {
    (void)params;
    fprintf(stderr, "[MOCK_CUPTI] cuptiPCSamplingDisable()\n");
    return CUPTI_SUCCESS;
}

CUptiResult cuptiPCSamplingStart(CUpti_PCSamplingStartParams *params) {
    (void)params;
    __pc_sampling_started = 1;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiPCSamplingStop(CUpti_PCSamplingStopParams *params) {
    (void)params;
    __pc_sampling_started = 0;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiPCSamplingSetConfigurationAttribute(
    CUpti_PCSamplingConfigurationInfoParams *params) {
    (void)params;
    fprintf(stderr, "[MOCK_CUPTI] cuptiPCSamplingSetConfigurationAttribute()\n");
    return CUPTI_SUCCESS;
}

CUptiResult cuptiPCSamplingGetNumStallReasons(
    CUpti_PCSamplingGetNumStallReasonsParams *params) {
    *params->numStallReasons = 3;
    return CUPTI_SUCCESS;
}

static const char *__mock_stall_names[] = {
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_barrier",
};

CUptiResult cuptiPCSamplingGetStallReasons(
    CUpti_PCSamplingGetStallReasonsParams *params) {
    size_t n = params->numStallReasons < 3 ? params->numStallReasons : 3;
    for (size_t i = 0; i < n; i++) {
        strncpy(params->stallReasons[i], __mock_stall_names[i],
                CUPTI_STALL_REASON_STRING_SIZE - 1);
        params->stallReasons[i][CUPTI_STALL_REASON_STRING_SIZE - 1] = '\0';
        params->stallReasonIndex[i] = (uint32_t)i;
    }
    return CUPTI_SUCCESS;
}

// ---------------------------------------------------------------------------
// Synthetic PC samples using real offsets from pc_sample_toy.cu kernels.
// Offsets extracted from nvdisasm -g on the sm_121 cubin.
// ---------------------------------------------------------------------------

typedef struct {
    const char *functionName;
    uint32_t functionIndex;
    uint64_t pcOffset;
    // Expected source correlation for this offset:
    uint32_t lineNumber;
    const char *fileName;
    const char *dirName;
} MockPCSample;

// Representative offsets covering all three kernels and distinct source lines.
// Offsets and line numbers from: nvdisasm -g -c pc_sample_toy.sm_121.cubin
static MockPCSample __mock_samples[] = {
    // shmem_bounce — shared-memory bouncing kernel
    {"_Z12shmem_bouncePfiy", 0, 0x00b0, 54,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // STS [R2], R3
    {"_Z12shmem_bouncePfiy", 0, 0x00d0, 55,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // BAR.SYNC
    {"_Z12shmem_bouncePfiy", 0, 0x01f0, 58,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // LDS (inner loop)
    {"_Z12shmem_bouncePfiy", 0, 0x0230, 58,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // FFMA (inner loop)
    {"_Z12shmem_bouncePfiy", 0, 0x0250, 59,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // BAR.SYNC (inner loop)

    // hash_churn — integer bit-twiddling kernel
    {"_Z10hash_churnPjiy", 1, 0x0050, 34,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // IMAD (idx calc)
    {"_Z10hash_churnPjiy", 1, 0x0080, 39,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // LDCU (loop start)
    {"_Z10hash_churnPjiy", 1, 0x0180, 40,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // IMAD.SHL (h ^= h << 13)

    // trig_storm — FP math kernel
    {"_Z10trig_stormPfiy", 2, 0x0050, 21,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // IMAD (idx calc)
    {"_Z10trig_stormPfiy", 2, 0x00b0, 25,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // FMUL (x * 0.001f)
    {"_Z10trig_stormPfiy", 2, 0x0f30, 27,
     "pc_sample_toy.cu", "/home/tpr/src/parcagpu-proton/microbenchmarks"},  // FMUL (sinf*cosf inner)
};
#define NUM_MOCK_SAMPLES (sizeof(__mock_samples) / sizeof(__mock_samples[0]))

static int __pc_get_data_calls = 0;
// Per-sample stall reason storage.  Allocated once and reused — pointers
// into this array are placed in each CUpti_PCSamplingPCData.
#define MAX_PC_DATA 16
static CUpti_PCSamplingStallReason __pc_stall_storage[MAX_PC_DATA][3];

// Most recent correlation ID from kernel launches.  The test harness
// updates this via __mock_pc_enqueue_correlation on every launch.
// cuptiPCSamplingGetData assigns recent IDs (counting back from the latest)
// so they overlap with traces still in the gpuTraceFixer's pcTraces map.
static volatile uint32_t __pc_latest_correlation = 0;

void __mock_pc_enqueue_correlation(uint32_t correlation_id) {
    __pc_latest_correlation = correlation_id;
}

CUptiResult cuptiPCSamplingGetData(CUpti_PCSamplingGetDataParams *params) {
    CUpti_PCSamplingData *data = params->pcSamplingData;

    if (data->collectNumPcs == 0) {
        data->totalNumPcs = 0;
        data->remainingNumPcs = 0;
        return CUPTI_SUCCESS;
    }

    // Cycle through the sample table, emitting a batch each time.
    static size_t sample_cursor = 0;
    size_t count = data->collectNumPcs < 4 ? data->collectNumPcs : 4;
    if (count > NUM_MOCK_SAMPLES)
        count = NUM_MOCK_SAMPLES;
    if (count > MAX_PC_DATA)
        count = MAX_PC_DATA;

    data->totalNumPcs = count;
    data->remainingNumPcs = 0;
    data->totalSamples = count * 9;

    for (size_t i = 0; i < count; i++) {
        MockPCSample *s = &__mock_samples[(sample_cursor + i) % NUM_MOCK_SAMPLES];
        CUpti_PCSamplingPCData *pc = &data->pPcData[i];
        pc->size = sizeof(CUpti_PCSamplingPCData);
        pc->cubinCrc = __cubin_crc;
        pc->pcOffset = s->pcOffset;
        pc->functionIndex = s->functionIndex;
        pc->functionName = (char *)s->functionName;
        pc->stallReasonCount = 3;
        pc->stallReason = __pc_stall_storage[i];
        for (size_t j = 0; j < 3; j++) {
            pc->stallReason[j].pcSamplingStallReasonIndex = (uint32_t)j;
            pc->stallReason[j].samples = (uint32_t)(5 - j * 2);
        }
        // Assign a correlation ID from recent-but-not-too-recent launches.
        // Offset by 20 from the latest to ensure the cuda_correlation BPF
        // probe has had time to fire and the trace is in pcTraces.
        uint32_t latest = __pc_latest_correlation;
        uint32_t back = 20 + (uint32_t)(count - 1 - i);
        pc->correlationId = (latest > back) ? latest - back : 0;
    }
    sample_cursor = (sample_cursor + count) % NUM_MOCK_SAMPLES;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiGetCubinCrc(CUpti_GetCubinCrcParams *params) {
    params->cubinCrc = __compute_crc(params->cubin, params->cubinSize);
    return CUPTI_SUCCESS;
}

// Source correlation: look up the offset in our known table.
// Falls back to zeros if the offset isn't in the table (same as real CUPTI
// when debug info is missing).
CUptiResult cuptiGetSassToSourceCorrelation(
    CUpti_GetSassToSourceCorrelationParams *params) {
    for (size_t i = 0; i < NUM_MOCK_SAMPLES; i++) {
        MockPCSample *s = &__mock_samples[i];
        if (params->pcOffset == s->pcOffset &&
            params->functionName && strcmp(params->functionName, s->functionName) == 0) {
            params->lineNumber = s->lineNumber;
            params->fileName = strdup(s->fileName);
            params->dirName = strdup(s->dirName);
            return CUPTI_SUCCESS;
        }
    }
    // Unknown offset — no source info available.
    params->lineNumber = 0;
    params->fileName = NULL;
    params->dirName = NULL;
    return CUPTI_SUCCESS;
}

// =========================================================================
// Resource callback helper — called from test harness after init.
// Fires CONTEXT_CREATED and MODULE_LOADED with the real cubin.
// =========================================================================

void __mock_cupti_fire_resource_callbacks(void) {
    if (!__cupti_runtime_api_callback)
        return;

    __load_cubin();

    fprintf(stderr, "[MOCK_CUPTI] Firing resource callbacks\n");

    // 1. CONTEXT_CREATED
    CUpti_ResourceData resData;
    memset(&resData, 0, sizeof(resData));
    resData.context = (CUcontext)(uintptr_t)0x1;
    __cupti_runtime_api_callback(
        __cupti_runtime_api_userdata,
        CUPTI_CB_DOMAIN_RESOURCE,
        CUPTI_CBID_RESOURCE_CONTEXT_CREATED,
        &resData);

    // 2. MODULE_LOADED with the real cubin.
    CUpti_ModuleResourceData modData;
    memset(&modData, 0, sizeof(modData));
    if (__cubin_data) {
        modData.pCubin = __cubin_data;
        modData.cubinSize = __cubin_size;
    } else {
        // Fallback: minimal fake cubin if no file was loaded.
        static const char fake[] = {0x7f, 'E', 'L', 'F', 0,0,0,0, 0,0,0,0, 0,0,0,0};
        modData.pCubin = fake;
        modData.cubinSize = sizeof(fake);
    }
    resData.resourceDescriptor = &modData;
    __cupti_runtime_api_callback(
        __cupti_runtime_api_userdata,
        CUPTI_CB_DOMAIN_RESOURCE,
        CUPTI_CBID_RESOURCE_MODULE_LOADED,
        &resData);
}

CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
    (void)kind;
    return CUPTI_SUCCESS;
}
