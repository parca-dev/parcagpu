#define _POSIX_C_SOURCE 199309L
#define _XOPEN_SOURCE 600
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <cuda.h>
#include <cupti.h>

//=============================================================================
// Configuration
//=============================================================================

typedef struct {
    int threads;              // Number of application threads
    int launch_rate;          // Kernel launches per second per GPU
    int graph_rate;           // Graph launches per second per GPU
    int num_gpus;             // Number of GPUs to simulate
    const char *kernel_names; // Path to kernel names file
    uint64_t duration;        // Run for N seconds (default: 5)
} TestConfig;

void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <library_path> [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --threads=N           Number of application threads (default: 1)\n");
    fprintf(stderr, "  --launch-rate=N       Kernel launches per second per GPU (default: 1000)\n");
    fprintf(stderr, "  --graph-rate=N        Graph launches per second per GPU (default: 0)\n");
    fprintf(stderr, "  --num-gpus=N          Number of GPUs to simulate (default: 1)\n");
    fprintf(stderr, "  --kernel-names=FILE   Path to kernel names file (default: generated)\n");
    fprintf(stderr, "  --duration=N          Run for N seconds (default: 5)\n");
    fprintf(stderr, "  --help, -h            Show this help message\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  # Simple test (backward compatible)\n");
    fprintf(stderr, "  %s ./libparcagpucupti.so\n\n", prog);
    fprintf(stderr, "  # 8 GPUs at 2500 launches/s per GPU for 10 seconds\n");
    fprintf(stderr, "  %s ./libparcagpucupti.so --num-gpus=8 --launch-rate=2500 --duration=10\n\n", prog);
    fprintf(stderr, "  # Multi-threaded test\n");
    fprintf(stderr, "  %s ./libparcagpucupti.so --threads=4 --num-gpus=8 --launch-rate=2500 --duration=10\n\n", prog);
}

TestConfig parse_args(int argc, char **argv, const char **lib_path) {
    TestConfig config = {
        .threads = 1,
        .launch_rate = 1000,
        .graph_rate = 0,
        .num_gpus = 1,
        .kernel_names = NULL,
        .duration = 5
    };

    // Check for help first (can be first arg)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }

    // First arg is library path (required)
    *lib_path = argc > 1 ? argv[1] : "./libparcagpucupti.so";

    // Parse remaining args starting from index 2
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--threads=", 10) == 0) {
            config.threads = atoi(argv[i] + 10);
        } else if (strncmp(argv[i], "--launch-rate=", 14) == 0) {
            config.launch_rate = atoi(argv[i] + 14);
        } else if (strncmp(argv[i], "--graph-rate=", 13) == 0) {
            config.graph_rate = atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--num-gpus=", 11) == 0) {
            config.num_gpus = atoi(argv[i] + 11);
        } else if (strncmp(argv[i], "--kernel-names=", 15) == 0) {
            config.kernel_names = argv[i] + 15;
        } else if (strncmp(argv[i], "--duration=", 11) == 0) {
            config.duration = atoi(argv[i] + 11);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    return config;
}

//=============================================================================
// Kernel Names
//=============================================================================

typedef struct {
    char **names;
    size_t count;
    atomic_size_t next_index;
} KernelNameList;

KernelNameList *load_kernel_names(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "Failed to open kernel names file: %s\n", filepath);
        return NULL;
    }

    KernelNameList *list = malloc(sizeof(KernelNameList));
    list->names = NULL;
    list->count = 0;
    atomic_init(&list->next_index, 0);

    char line[256];
    size_t capacity = 16;
    list->names = malloc(capacity * sizeof(char *));

    while (fgets(line, sizeof(line), f)) {
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }

        if (strlen(line) == 0) continue;

        if (list->count >= capacity) {
            capacity *= 2;
            list->names = realloc(list->names, capacity * sizeof(char *));
        }

        list->names[list->count++] = strdup(line);
    }

    fclose(f);

    if (list->count == 0) {
        free(list->names);
        free(list);
        return NULL;
    }

    fprintf(stderr, "Loaded %zu kernel names from %s\n", list->count, filepath);
    return list;
}

const char *get_next_kernel_name(KernelNameList *list) {
    if (!list || list->count == 0) {
        return "mock_kernel";
    }

    size_t idx = atomic_fetch_add(&list->next_index, 1) % list->count;
    return list->names[idx];
}

void free_kernel_names(KernelNameList *list) {
    if (!list) return;

    for (size_t i = 0; i < list->count; i++) {
        free(list->names[i]);
    }
    free(list->names);
    free(list);
}

//=============================================================================
// Timestamp Generation
//=============================================================================

static inline uint64_t get_current_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline uint64_t generate_kernel_duration_ns(int launch_rate_per_gpu) {
    uint64_t max_duration_ns = 1000000000ULL / launch_rate_per_gpu;
    uint64_t min_duration_ns = 5000ULL; // 5μs
    if (max_duration_ns <= min_duration_ns) {
        return min_duration_ns;
    }
    // Random duration between 5μs and max
    return min_duration_ns + (rand() % (max_duration_ns - min_duration_ns));
}

//=============================================================================
// Global State
//=============================================================================

typedef int (*InitializeInjectionFunc)(void);

// Callback pointers (set by InitializeInjection)
static void (*bufferRequestedCallback)(uint8_t **buffer, size_t *size, size_t *maxNumRecords) = NULL;
static void (*bufferCompletedCallback)(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) = NULL;
static void (*parcagpuCuptiCallback)(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata) = NULL;

//=============================================================================
// Launched Kernels Queue
// Track which correlation IDs have had their callbacks executed
//=============================================================================

#define MAX_PENDING_ACTIVITIES 100000

typedef struct {
    uint32_t correlation_ids[MAX_PENDING_ACTIVITIES];
    size_t write_idx;
    size_t read_idx;
    pthread_mutex_t mutex;
} LaunchedKernelsQueue;

static LaunchedKernelsQueue launched_queue;

void init_launched_queue(void) {
    launched_queue.write_idx = 0;
    launched_queue.read_idx = 0;
    pthread_mutex_init(&launched_queue.mutex, NULL);
}

void enqueue_launched_kernel(uint32_t correlation_id) {
    pthread_mutex_lock(&launched_queue.mutex);
    size_t next_write = (launched_queue.write_idx + 1) % MAX_PENDING_ACTIVITIES;
    if (next_write != launched_queue.read_idx) {
        launched_queue.correlation_ids[launched_queue.write_idx] = correlation_id;
        launched_queue.write_idx = next_write;
    } else {
        fprintf(stderr, "WARNING: Launched kernels queue full, dropping correlation ID %u\n", correlation_id);
    }
    pthread_mutex_unlock(&launched_queue.mutex);
}

bool dequeue_launched_kernel(uint32_t *correlation_id) {
    pthread_mutex_lock(&launched_queue.mutex);
    if (launched_queue.read_idx == launched_queue.write_idx) {
        pthread_mutex_unlock(&launched_queue.mutex);
        return false;
    }
    *correlation_id = launched_queue.correlation_ids[launched_queue.read_idx];
    launched_queue.read_idx = (launched_queue.read_idx + 1) % MAX_PENDING_ACTIVITIES;
    pthread_mutex_unlock(&launched_queue.mutex);
    return true;
}

size_t get_queue_size(void) {
    pthread_mutex_lock(&launched_queue.mutex);
    size_t size;
    if (launched_queue.write_idx >= launched_queue.read_idx) {
        size = launched_queue.write_idx - launched_queue.read_idx;
    } else {
        size = MAX_PENDING_ACTIVITIES - launched_queue.read_idx + launched_queue.write_idx;
    }
    pthread_mutex_unlock(&launched_queue.mutex);
    return size;
}

//=============================================================================
// Callback Simulation
//=============================================================================

void simulate_runtime_kernel_launch(uint32_t correlationId, CUpti_CallbackId cbid) {
    CUpti_CallbackData cbdata = {0};

    if (!parcagpuCuptiCallback) {
        fprintf(stderr, "ERROR: parcagpuCuptiCallback is NULL!\n");
        return;
    }

    // RUNTIME ENTER callback
    cbdata.callbackSite = CUPTI_API_ENTER;
    cbdata.correlationId = correlationId;
    parcagpuCuptiCallback(NULL, CUPTI_CB_DOMAIN_RUNTIME_API, cbid, &cbdata);

    // DRIVER ENTER callback (runtime internally calls driver)
    cbdata.callbackSite = CUPTI_API_ENTER;
    parcagpuCuptiCallback(NULL, CUPTI_CB_DOMAIN_DRIVER_API,
                         CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, &cbdata);

    // DRIVER EXIT callback
    cbdata.callbackSite = CUPTI_API_EXIT;
    parcagpuCuptiCallback(NULL, CUPTI_CB_DOMAIN_DRIVER_API,
                         CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, &cbdata);

    // RUNTIME EXIT callback
    cbdata.callbackSite = CUPTI_API_EXIT;
    parcagpuCuptiCallback(NULL, CUPTI_CB_DOMAIN_RUNTIME_API, cbid, &cbdata);

    // Enqueue this correlation ID for activity generation
    enqueue_launched_kernel(correlationId);
}

void simulate_driver_kernel_launch(uint32_t correlationId, CUpti_CallbackId cbid) {
    CUpti_CallbackData cbdata = {0};

    if (!parcagpuCuptiCallback) {
        fprintf(stderr, "ERROR: parcagpuCuptiCallback is NULL!\n");
        return;
    }

    // DRIVER ENTER callback
    cbdata.callbackSite = CUPTI_API_ENTER;
    cbdata.correlationId = correlationId;
    parcagpuCuptiCallback(NULL, CUPTI_CB_DOMAIN_DRIVER_API, cbid, &cbdata);

    // DRIVER EXIT callback
    cbdata.callbackSite = CUPTI_API_EXIT;
    parcagpuCuptiCallback(NULL, CUPTI_CB_DOMAIN_DRIVER_API, cbid, &cbdata);

    // Enqueue this correlation ID for activity generation
    enqueue_launched_kernel(correlationId);
}

//=============================================================================
// Multi-threaded Test Infrastructure
//=============================================================================

typedef struct {
    TestConfig *config;
    KernelNameList *kernel_names;
    int thread_id;
    atomic_uint_least32_t *global_correlation_id;
    atomic_bool *should_stop;
    pthread_barrier_t *start_barrier;
} WorkerThreadArgs;

// Worker thread simulates kernel launches
void *worker_thread(void *arg) {
    WorkerThreadArgs *args = (WorkerThreadArgs *)arg;

    // Wait for all threads to be ready
    pthread_barrier_wait(args->start_barrier);

    uint64_t sleep_ns = 1000000000ULL / args->config->launch_rate;
    struct timespec sleep_time = {0, (long)sleep_ns};

    uint64_t iterations = 0;
    uint64_t max_iterations = args->config->duration * args->config->launch_rate;

    while (!atomic_load(args->should_stop) && iterations < max_iterations) {
        // Generate 5 kernel launches per iteration
        for (int j = 0; j < 5; j++) {
            uint32_t correlationId = atomic_fetch_add(args->global_correlation_id, 1);
            bool is_graph = (args->config->graph_rate > 0 && correlationId % 3 == 0);
            bool is_runtime = (correlationId % 2 == 0); // 50% runtime, 50% driver

            if (is_graph) {
                if (is_runtime) {
                    simulate_runtime_kernel_launch(correlationId, CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000);
                } else {
                    simulate_driver_kernel_launch(correlationId, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch);
                }
            } else {
                if (is_runtime) {
                    simulate_runtime_kernel_launch(correlationId, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
                } else {
                    simulate_driver_kernel_launch(correlationId, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
                }
            }
        }

        nanosleep(&sleep_time, NULL);
        iterations++;
    }

    return NULL;
}

// Dedicated CUPTI thread processes activity buffers
// Simulates CUPTI's periodic flush behavior by batching multiple activities into buffers
// Only generates activities for correlation IDs that have had callbacks executed
void *cupti_thread(void *arg) {
    WorkerThreadArgs *args = (WorkerThreadArgs *)arg;

    // Wait for all threads to be ready
    pthread_barrier_wait(args->start_barrier);

    uint32_t graphId = 1000;
    struct timespec sleep_time = {0, 10000000}; // 10ms - simulates flush period

    while (!atomic_load(args->should_stop)) {
        size_t queue_size = get_queue_size();

        // Only flush if we have pending activities
        if (queue_size > 0 && bufferRequestedCallback && bufferCompletedCallback) {
            // Request a buffer from the profiler (simulates CUPTI requesting a buffer)
            uint8_t *buffer;
            size_t bufferSize;
            size_t maxNumRecords;
            bufferRequestedCallback(&buffer, &bufferSize, &maxNumRecords);

            // Fill the buffer with activity records for launched kernels
            size_t offset = 0;
            size_t recordSize = sizeof(CUpti_ActivityKernel5);

            // Dequeue and process as many launched kernels as will fit in the buffer
            uint32_t correlationId;
            while (dequeue_launched_kernel(&correlationId) &&
                   offset + recordSize <= bufferSize &&
                   !atomic_load(args->should_stop)) {

                uint64_t now = get_current_time_ns();
                uint64_t duration = generate_kernel_duration_ns(args->config->launch_rate);
                uint32_t gpu_id = correlationId % args->config->num_gpus;

                // Create activity record directly in the buffer
                CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *)(buffer + offset);
                kernel->kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
                kernel->correlationId = correlationId;
                kernel->deviceId = gpu_id;
                kernel->streamId = 1;
                kernel->start = now;
                kernel->end = now + duration;

                bool is_graph = (args->config->graph_rate > 0 && correlationId % 3 == 0);
                if (is_graph) {
                    kernel->graphId = graphId++;
                    kernel->graphNodeId = 100;
                    kernel->name = "graph_kernel";
                } else {
                    kernel->graphId = 0;
                    kernel->graphNodeId = 0;
                    kernel->name = get_next_kernel_name(args->kernel_names);
                    if (!kernel->name) kernel->name = "mock_kernel";
                }

                offset += recordSize;
            }

            // Complete the buffer (simulates CUPTI calling the completion callback)
            if (offset > 0) {
                bufferCompletedCallback(NULL, 1, buffer, bufferSize, offset);
            }
        }

        nanosleep(&sleep_time, NULL);
    }

    return NULL;
}

//=============================================================================
// Test Runner
//=============================================================================

void run_test(TestConfig *config, KernelNameList *kernel_names) {
    fprintf(stderr, "\n=== Starting test ===\n");
    fprintf(stderr, "Threads: %d worker%s + 1 CUPTI thread\n",
            config->threads, config->threads == 1 ? "" : "s");
    fprintf(stderr, "Total launch rate: %d launches/s (%d per worker)\n",
            config->threads * config->launch_rate, config->launch_rate);
    fprintf(stderr, "GPUs: %d\n", config->num_gpus);
    fprintf(stderr, "Duration: %lu seconds\n", config->duration);

    // Initialize the launched kernels queue
    init_launched_queue();

    atomic_uint_least32_t global_correlation_id;
    atomic_init(&global_correlation_id, 1);

    atomic_bool should_stop;
    atomic_init(&should_stop, false);

    // Create barrier for synchronized start (workers + CUPTI thread)
    pthread_barrier_t start_barrier;
    pthread_barrier_init(&start_barrier, NULL, config->threads + 1);

    // Create worker threads
    WorkerThreadArgs *worker_args = malloc(config->threads * sizeof(WorkerThreadArgs));
    pthread_t *worker_threads = malloc(config->threads * sizeof(pthread_t));

    for (int i = 0; i < config->threads; i++) {
        worker_args[i].config = config;
        worker_args[i].kernel_names = kernel_names;
        worker_args[i].thread_id = i;
        worker_args[i].global_correlation_id = &global_correlation_id;
        worker_args[i].should_stop = &should_stop;
        worker_args[i].start_barrier = &start_barrier;

        if (pthread_create(&worker_threads[i], NULL, worker_thread, &worker_args[i]) != 0) {
            fprintf(stderr, "Failed to create worker thread %d\n", i);
            atomic_store(&should_stop, true);
            for (int j = 0; j < i; j++) {
                pthread_join(worker_threads[j], NULL);
            }
            free(worker_args);
            free(worker_threads);
            pthread_barrier_destroy(&start_barrier);
            return;
        }
    }

    // Create CUPTI thread
    pthread_t cupti_pthread;
    WorkerThreadArgs cupti_args = {
        .config = config,
        .kernel_names = kernel_names,
        .thread_id = -1,
        .global_correlation_id = &global_correlation_id,
        .should_stop = &should_stop,
        .start_barrier = &start_barrier
    };

    if (pthread_create(&cupti_pthread, NULL, cupti_thread, &cupti_args) != 0) {
        fprintf(stderr, "Failed to create CUPTI thread\n");
        atomic_store(&should_stop, true);
        for (int i = 0; i < config->threads; i++) {
            pthread_join(worker_threads[i], NULL);
        }
        free(worker_args);
        free(worker_threads);
        pthread_barrier_destroy(&start_barrier);
        return;
    }

    fprintf(stderr, "All threads started, running for %lu seconds...\n", config->duration);

    // Wait for duration then signal stop
    sleep(config->duration);
    atomic_store(&should_stop, true);

    // Wait for all threads to complete
    for (int i = 0; i < config->threads; i++) {
        pthread_join(worker_threads[i], NULL);
    }
    pthread_join(cupti_pthread, NULL);

    uint32_t total_events = atomic_load(&global_correlation_id) - 1;
    fprintf(stderr, "Test completed. Generated %u events\n", total_events);

    free(worker_args);
    free(worker_threads);
    pthread_barrier_destroy(&start_barrier);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char **argv) {
    const char *lib_path;
    TestConfig config = parse_args(argc, argv, &lib_path);

    fprintf(stderr, "Loading library: %s\n", lib_path);
    fprintf(stderr, "Configuration:\n");
    fprintf(stderr, "  Threads: %d\n", config.threads);
    fprintf(stderr, "  Launch rate: %d/s per GPU\n", config.launch_rate);
    fprintf(stderr, "  Graph rate: %d/s per GPU\n", config.graph_rate);
    fprintf(stderr, "  Num GPUs: %d\n", config.num_gpus);
    fprintf(stderr, "  Duration: %lu seconds\n", config.duration);
    if (config.kernel_names) {
        fprintf(stderr, "  Kernel names: %s\n", config.kernel_names);
    }

    void *cupti_prof_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!cupti_prof_handle) {
        fprintf(stderr, "Failed to load library: %s\n", dlerror());
        return 1;
    }

    InitializeInjectionFunc initFunc = (InitializeInjectionFunc)dlsym(cupti_prof_handle, "InitializeInjection");
    if (!initFunc) {
        fprintf(stderr, "Failed to find InitializeInjection: %s\n", dlerror());
        dlclose(cupti_prof_handle);
        return 1;
    }

    fprintf(stderr, "Calling InitializeInjection...\n");
    int result = initFunc();
    fprintf(stderr, "InitializeInjection returned: %d\n", result);

    // Get callback pointers from mock CUPTI
    void **runtime_api_cb_ptr = (void **)dlsym(RTLD_DEFAULT, "__cupti_runtime_api_callback");
    void **buffer_requested_cb_ptr = (void **)dlsym(RTLD_DEFAULT, "__cupti_buffer_requested_callback");
    void **buffer_completed_cb_ptr = (void **)dlsym(RTLD_DEFAULT, "__cupti_buffer_completed_callback");

    if (runtime_api_cb_ptr) {
        parcagpuCuptiCallback = (void (*)(void *, CUpti_CallbackDomain, CUpti_CallbackId, const CUpti_CallbackData *))*runtime_api_cb_ptr;
    }
    if (buffer_requested_cb_ptr) {
        bufferRequestedCallback = (void (*)(uint8_t **, size_t *, size_t *))*buffer_requested_cb_ptr;
    }
    if (buffer_completed_cb_ptr) {
        bufferCompletedCallback = (void (*)(CUcontext, uint32_t, uint8_t *, size_t, size_t))*buffer_completed_cb_ptr;
    }

    if (!parcagpuCuptiCallback || !bufferCompletedCallback) {
        fprintf(stderr, "Warning: Could not get callback pointers from mock CUPTI.\n");
        dlclose(cupti_prof_handle);
        return 0;
    }

    // Load kernel names if specified
    KernelNameList *kernel_names = NULL;
    if (config.kernel_names) {
        kernel_names = load_kernel_names(config.kernel_names);
        if (!kernel_names) {
            fprintf(stderr, "Warning: Failed to load kernel names, using generated names\n");
        }
    }

    // Initialize random seed for duration generation
    srand(time(NULL));

    // Run test
    run_test(&config, kernel_names);

    // Cleanup
    free_kernel_names(kernel_names);

    typedef void (*CleanupFunc)(void);
    CleanupFunc cleanup = (CleanupFunc)dlsym(cupti_prof_handle, "cleanup");
    if (cleanup) {
        cleanup();
    }

    if (runtime_api_cb_ptr) *runtime_api_cb_ptr = NULL;
    if (buffer_requested_cb_ptr) *buffer_requested_cb_ptr = NULL;
    if (buffer_completed_cb_ptr) *buffer_completed_cb_ptr = NULL;

    dlclose(cupti_prof_handle);
    fprintf(stderr, "Cleanup complete.\n");

    _exit(0);
}
