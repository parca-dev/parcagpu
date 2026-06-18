// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// Focused unit test for NVTX callback handling in libparcagpucupti.so.
// The library subscribes to CUPTI_CB_DOMAIN_NVTX during InitializeInjection;
// we then synthesize NVTX cbdata and dispatch it via the registered callback
// (resolved through the mock CUPTI's __cupti_runtime_api_callback global),
// and assert via the test-only `parcagpu_nvtx_test_*` accessors that the
// collector captured / batched / named as expected.
//
// No GPU, no live USDT consumer. The collector's `force_capture` test hook
// makes the early-exit (semaphore-cold) path always proceed, so we can
// observe behavior in this environment.

#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cupti.h>
// cupti.h transitively includes cupti_nvtx_cbid.h on CUDA 12.x.

// ---- NVTX param struct shadows (copied from generated_nvtx_meta.h) ------
// The library's collector reads these layouts; we synthesize them here.

typedef struct {
  const char *message;
} nvtx_RangePushA_params;
typedef struct {
  void *dummy;
} nvtx_RangePop_params;
typedef struct {
  const char *message;
} nvtx_MarkA_params;
typedef struct {
  const char *message;
} nvtx_RangeStartA_params;
typedef struct {
  uint64_t id;
} nvtx_RangeEnd_params;
typedef struct {
  uint32_t threadId;
  const char *name;
} nvtx_NameOsThreadA_params;
typedef struct {
  void *stream;
  const char *name;
} nvtx_NameCudaStreamA_params;

// Same layout as the collector's nvtxEventAttributes_v2 shadow.
typedef struct {
  uint16_t version;
  uint16_t size;
  uint32_t category;
  int32_t  colorType;
  uint32_t color;
  int32_t  payloadType;
  int32_t  reserved0;
  union { uint64_t ull; int64_t ll; double d; uint32_t ui; int32_t i; float f; } payload;
  int32_t  messageType;
  union { const char *ascii; const void *wide; } message;
} nvtx_EventAttribs;

// ---- Function-pointer types for the dispatched callback + accessors -----

typedef void (*CuptiCb_t)(void *, CUpti_CallbackDomain, CUpti_CallbackId,
                          const void *);
typedef int (*InitFn_t)(void);
typedef void (*ForceCapFn_t)(int);
typedef uint64_t (*StatFn_t)(void);
typedef void (*ResetFn_t)(void);

static void fail(const char *what) {
  fprintf(stderr, "FAIL: %s\n", what);
  exit(1);
}

static void fire(CuptiCb_t cb, void *ud, CUpti_CallbackId cbid,
                 const void *params, const void *retval) {
  CUpti_NvtxData nd;
  memset(&nd, 0, sizeof(nd));
  nd.functionName = "(test)";
  nd.functionParams = params;
  nd.functionReturnValue = retval;
  cb(ud, CUPTI_CB_DOMAIN_NVTX, cbid, &nd);
}

int main(void) {
  const char *libpath = getenv("PARCAGPU_TEST_LIB");
  if (!libpath)
    libpath = "./libparcagpucupti.so";

  // Disable PC sampling so the lib doesn't try to drive its sampling state
  // machine against the mock CUPTI in ways unrelated to NVTX.
  setenv("PARCAGPU_PC_SAMPLING_RATE", "0", 1);

  void *h = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL);
  if (!h)
    fail(dlerror());

  InitFn_t init = (InitFn_t)dlsym(h, "InitializeInjection");
  if (!init)
    fail("missing InitializeInjection");
  if (init() != 1)
    fail("InitializeInjection returned non-1");

  // Resolve the mock CUPTI's stored callback + userdata; the lib registered
  // these via cuptiSubscribe during init.
  void **cb_slot = (void **)dlsym(RTLD_DEFAULT, "__cupti_runtime_api_callback");
  void **ud_slot = (void **)dlsym(RTLD_DEFAULT, "__cupti_runtime_api_userdata");
  if (!cb_slot || !*cb_slot)
    fail("mock callback not registered");
  CuptiCb_t cb = (CuptiCb_t)*cb_slot;
  void *ud = ud_slot ? *ud_slot : NULL;

  // Test hooks.
  ForceCapFn_t force =
      (ForceCapFn_t)dlsym(h, "parcagpu_nvtx_test_force_capture");
  StatFn_t events  = (StatFn_t)dlsym(h, "parcagpu_nvtx_test_events_captured");
  StatFn_t batches = (StatFn_t)dlsym(h, "parcagpu_nvtx_test_batches_fired");
  StatFn_t names   = (StatFn_t)dlsym(h, "parcagpu_nvtx_test_resource_names_fired");
  ResetFn_t reset  = (ResetFn_t)dlsym(h, "parcagpu_nvtx_test_reset");
  if (!force || !events || !batches || !names || !reset)
    fail("missing test accessors");

  force(1);

  // -------- Case 1: stacked push/pop + mark --------
  reset();
  nvtx_RangePushA_params push = { "iter_0" };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxRangePushA, &push, NULL);
  nvtx_MarkA_params mark = { "checkpoint" };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxMarkA, &mark, NULL);
  nvtx_RangePop_params pop = { NULL };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxRangePop, &pop, NULL);
  if (events() != 3) { fprintf(stderr, "events=%lu want=3\n", events()); fail("case1 count"); }
  if (batches() != 0) { fail("case1 should not have fired yet"); }

  // -------- Case 2: non-stacked start/end with returned rangeId --------
  reset();
  nvtx_RangeStartA_params start = { "phase_A" };
  uint64_t rangeId = 0x1122334455667788ULL;
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxRangeStartA, &start, &rangeId);
  nvtx_RangeEnd_params end = { 0x1122334455667788ULL };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxRangeEnd, &end, NULL);
  if (events() != 2) { fail("case2 count"); }

  // -------- Case 3: Ex variant with color + payload --------
  reset();
  nvtx_EventAttribs attr;
  memset(&attr, 0, sizeof(attr));
  attr.version    = 2;
  attr.size       = sizeof(attr);
  attr.colorType  = 1;  // NVTX_COLOR_ARGB
  attr.color      = 0xff112233u;
  attr.payload.ull = 0xdeadbeefcafef00dULL;
  attr.messageType = 1; // NVTX_MESSAGE_TYPE_ASCII
  attr.message.ascii = "ex_event";
  struct { const void *eventAttrib; } ex = { &attr };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxRangePushEx, &ex, NULL);
  if (events() != 1) { fail("case3 push count"); }

  // -------- Case 4: resource naming fires nvtx_resource_name --------
  reset();
  nvtx_NameOsThreadA_params name_thr = { 1234, "worker" };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxNameOsThreadA, &name_thr, NULL);
  nvtx_NameCudaStreamA_params name_stream =
      { (void *)(uintptr_t)0xabc, "compute_stream" };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxNameCudaStreamA, &name_stream, NULL);
  if (names() != 2) { fprintf(stderr, "names=%lu want=2\n", names()); fail("case4 names"); }
  if (events() != 0) { fail("case4 should not capture events"); }

  // -------- Case 5: batch fires when full --------
  reset();
  // The default batch size from CMakeLists.txt is 256. Push that many
  // events on this thread; that should fire exactly one batch.
  for (int i = 0; i < 256; ++i) {
    nvtx_MarkA_params m = { "x" };
    fire(cb, ud, CUPTI_CBID_NVTX_nvtxMarkA, &m, NULL);
  }
  if (events() != 256) { fail("case5 events"); }
  if (batches() != 1)  { fprintf(stderr, "batches=%lu want=1\n", batches()); fail("case5 batches"); }
  // One more event — should sit in a new batch, not fire yet.
  nvtx_MarkA_params one = { "tail" };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxMarkA, &one, NULL);
  if (events() != 257) { fail("case5 tail event"); }
  if (batches() != 1)  { fail("case5 should still be 1 batch"); }

  // -------- Case 6: force-capture off restores cold path --------
  force(0);
  reset();
  nvtx_RangePushA_params hot = { "skipped" };
  fire(cb, ud, CUPTI_CBID_NVTX_nvtxRangePushA, &hot, NULL);
  // No USDT consumer attached, so without force-capture nothing should
  // run past the early exit.
  if (events() != 0) { fprintf(stderr, "events=%lu want=0\n", events()); fail("case6 cold path"); }

  fprintf(stderr, "PASS\n");
  return 0;
}
