// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// Focused unit test for the NVTX injection in libparcagpucupti.so. The test
// plays the part of an NVTX v3 client copy: it builds the per-module function
// tables a client exposes through its export table, calls
// InitializeInjectionNvtx2, and then calls through whatever the library
// installed. Assertions go through the test-only `parcagpu_nvtx_test_*`
// accessors.
//
// No GPU, no live USDT consumer. The collector's `force_capture` test hook
// makes the early-exit (semaphore-cold) path always proceed, so we can
// observe behavior in this environment.

#define _GNU_SOURCE // dladdr
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NVTX_NO_IMPL
#include <nvtx3/nvToolsExt.h>

#include "../src/nvtx.h"

// ---- A fake NVTX client ----------------------------------------------------

typedef struct {
  NvtxFunctionPointer core[NVTX_CBID_CORE_SIZE];
  NvtxFunctionPointer core2[NVTX_CBID_CORE2_SIZE];
  NvtxFunctionPointer cuda[NVTX_CBID_CUDA_SIZE];
  NvtxFunctionPointer cudart[NVTX_CBID_CUDART_SIZE];
  NvtxFunctionPointer *coreTbl[NVTX_CBID_CORE_SIZE];
  NvtxFunctionPointer *core2Tbl[NVTX_CBID_CORE2_SIZE];
  NvtxFunctionPointer *cudaTbl[NVTX_CBID_CUDA_SIZE];
  NvtxFunctionPointer *cudartTbl[NVTX_CBID_CUDART_SIZE];
} Client;

// GetModuleFunctionTable has no context argument, so the export table of
// whichever client is initializing is selected through this global.
static Client *g_initClient;

static int getModuleFunctionTable(NvtxCallbackModule m, NvtxFunctionTable *out,
                                  unsigned int *size) {
  Client *c = g_initClient;
  switch (m) {
  case NVTX_CB_MODULE_CORE:   *out = c->coreTbl;   *size = NVTX_CBID_CORE_SIZE;   return 1;
  case NVTX_CB_MODULE_CORE2:  *out = c->core2Tbl;  *size = NVTX_CBID_CORE2_SIZE;  return 1;
  case NVTX_CB_MODULE_CUDA:   *out = c->cudaTbl;   *size = NVTX_CBID_CUDA_SIZE;   return 1;
  case NVTX_CB_MODULE_CUDART: *out = c->cudartTbl; *size = NVTX_CBID_CUDART_SIZE; return 1;
  default: return 0;
  }
}

static const NvtxExportTableCallbacks g_callbacks = {
    sizeof(NvtxExportTableCallbacks), getModuleFunctionTable};

static const void *getExportTable(uint32_t id) {
  return id == NVTX_ETID_CALLBACKS ? &g_callbacks : NULL;
}

static void clientInit(Client *c) {
  memset(c, 0, sizeof(*c));
  for (int i = 0; i < NVTX_CBID_CORE_SIZE; ++i) c->coreTbl[i] = &c->core[i];
  for (int i = 0; i < NVTX_CBID_CORE2_SIZE; ++i) c->core2Tbl[i] = &c->core2[i];
  for (int i = 0; i < NVTX_CBID_CUDA_SIZE; ++i) c->cudaTbl[i] = &c->cuda[i];
  for (int i = 0; i < NVTX_CBID_CUDART_SIZE; ++i) c->cudartTbl[i] = &c->cudart[i];
}

// Typed views of the installed slots.
typedef void (*MarkA_t)(const char *);
typedef int (*PushA_t)(const char *);
typedef int (*PushEx_t)(const nvtxEventAttributes_t *);
typedef int (*Pop_t)(void);
typedef nvtxRangeId_t (*StartA_t)(const char *);
typedef void (*End_t)(nvtxRangeId_t);
typedef void (*NameOsThreadA_t)(uint32_t, const char *);
typedef nvtxDomainHandle_t (*DomainCreateA_t)(const char *);
typedef nvtxStringHandle_t (*RegisterStringA_t)(nvtxDomainHandle_t, const char *);
typedef int (*DomainPushEx_t)(nvtxDomainHandle_t, const nvtxEventAttributes_t *);
typedef int (*DomainPop_t)(nvtxDomainHandle_t);
typedef void (*NameCudaStreamA_t)(void *, const char *);

// ---- Library hooks -----------------------------------------------------------

typedef int (*InitNvtx_t)(NvtxGetExportTableFunc_t);
typedef int (*InitFn_t)(void);
typedef void (*ForceCapFn_t)(int);
typedef uint64_t (*StatFn_t)(void);
typedef void (*ResetFn_t)(void);
typedef const char *(*LastEventFn_t)(struct nvtx_event *);

static void fail(const char *what) {
  fprintf(stderr, "FAIL: %s\n", what);
  exit(1);
}

#define EXPECT(cond, what)                                                     \
  do {                                                                         \
    if (!(cond))                                                               \
      fail(what);                                                              \
  } while (0)

int main(void) {
  const char *libpath = getenv("PARCAGPU_TEST_LIB");
  if (!libpath)
    libpath = "./libparcagpucupti.so";
  unsetenv("NVTX_INJECTION64_PATH");

  void *h = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL);
  if (!h)
    fail(dlerror());

  InitNvtx_t initNvtx = (InitNvtx_t)dlsym(h, "InitializeInjectionNvtx2");
  InitFn_t initCuda = (InitFn_t)dlsym(h, "InitializeInjection");
  ForceCapFn_t force =
      (ForceCapFn_t)dlsym(h, "parcagpu_nvtx_test_force_capture");
  StatFn_t events  = (StatFn_t)dlsym(h, "parcagpu_nvtx_test_events_captured");
  StatFn_t batches = (StatFn_t)dlsym(h, "parcagpu_nvtx_test_batches_fired");
  StatFn_t names   = (StatFn_t)dlsym(h, "parcagpu_nvtx_test_resource_names_fired");
  ResetFn_t reset  = (ResetFn_t)dlsym(h, "parcagpu_nvtx_test_reset");
  LastEventFn_t last = (LastEventFn_t)dlsym(h, "parcagpu_nvtx_test_last_event");
  if (!initNvtx || !initCuda || !force || !events || !batches || !names ||
      !reset || !last)
    fail("missing library entry points");

  force(1);

  // -------- Case 1: injection installs handlers into a client --------
  // Like an NVTX client at its first call, typically before cuInit.
  static Client a;
  clientInit(&a);
  g_initClient = &a;
  EXPECT(initNvtx(getExportTable) == 1, "InitializeInjectionNvtx2 != 1");
  EXPECT(a.core[NVTX_CBID_CORE_RangePushA] != NULL, "RangePushA not installed");
  EXPECT(a.core2[NVTX_CBID_CORE2_DomainRangePushEx] != NULL,
         "DomainRangePushEx not installed");
  EXPECT(a.core[NVTX_CBID_CORE_RangePushW] == NULL,
         "RangePushW should stay unassigned (no-op)");

  struct nvtx_event ev;
  const char *msg;

  // -------- Case 2: stacked push/pop + mark, before CUDA init --------
  reset();
  PushA_t pushA = (PushA_t)a.core[NVTX_CBID_CORE_RangePushA];
  Pop_t pop = (Pop_t)a.core[NVTX_CBID_CORE_RangePop];
  MarkA_t markA = (MarkA_t)a.core[NVTX_CBID_CORE_MarkA];
  EXPECT(pushA("iter_0") == 0, "first push level != 0");
  markA("checkpoint");
  msg = last(&ev);
  EXPECT(ev.kind == NVTX_KIND_MARK && strcmp(msg, "checkpoint") == 0,
         "mark not captured");
  EXPECT(ev.domain_id == 0, "core API should use default domain");
  EXPECT(ev.device_id == UINT32_MAX, "device resolved before cuInit");
  EXPECT(pop() == 0, "pop level != 0");
  EXPECT(events() == 3, "case2 count");
  EXPECT(batches() == 0, "case2 should not have fired yet");

  // -------- Case 3: non-stacked start/end, range ids from the injection ----
  reset();
  StartA_t startA = (StartA_t)a.core[NVTX_CBID_CORE_RangeStartA];
  End_t end = (End_t)a.core[NVTX_CBID_CORE_RangeEnd];
  nvtxRangeId_t r1 = startA("phase_A");
  nvtxRangeId_t r2 = startA("phase_B");
  EXPECT(r1 != 0 && r2 != 0 && r1 != r2, "range ids not unique");
  end(r1);
  last(&ev);
  EXPECT(ev.kind == NVTX_KIND_RANGE_END && ev.range_id == r1, "end range id");
  EXPECT(events() == 3, "case3 count");

  // -------- Case 4: Ex variant with color + payload --------
  reset();
  nvtxEventAttributes_t attr;
  memset(&attr, 0, sizeof(attr));
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.colorType = NVTX_COLOR_ARGB;
  attr.color = 0xff112233u;
  attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  attr.payload.ullValue = 0xdeadbeefcafef00dULL;
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = "ex_event";
  ((PushEx_t)a.core[NVTX_CBID_CORE_RangePushEx])(&attr);
  msg = last(&ev);
  EXPECT(ev.color == 0xff112233u, "ex color");
  EXPECT(ev.payload == 0xdeadbeefcafef00dULL, "ex payload");
  EXPECT(strcmp(msg, "ex_event") == 0, "ex message");
  pop();

  // -------- Case 5: domains + registered strings (Python nvtx's path) -------
  reset();
  DomainCreateA_t createA =
      (DomainCreateA_t)a.core2[NVTX_CBID_CORE2_DomainCreateA];
  RegisterStringA_t regA =
      (RegisterStringA_t)a.core2[NVTX_CBID_CORE2_DomainRegisterStringA];
  DomainPushEx_t dpush =
      (DomainPushEx_t)a.core2[NVTX_CBID_CORE2_DomainRangePushEx];
  DomainPop_t dpop = (DomainPop_t)a.core2[NVTX_CBID_CORE2_DomainRangePop];

  nvtxDomainHandle_t d = createA("TensorRT-LLM");
  EXPECT(d != NULL, "domain create");
  EXPECT(names() == 1, "domain name should be announced on create");
  nvtxStringHandle_t s = regA(d, "forward_step");
  EXPECT(s != NULL && regA(d, "forward_step") == s, "registered string dedup");

  memset(&attr, 0, sizeof(attr));
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attr.message.registered = s;
  dpush(d, &attr);
  msg = last(&ev);
  EXPECT(ev.kind == NVTX_KIND_RANGE_PUSH, "domain push kind");
  EXPECT(ev.domain_id != 0, "domain push should carry the domain id");
  EXPECT(strcmp(msg, "forward_step") == 0, "registered string not resolved");
  uint32_t trtDomain = ev.domain_id;
  dpop(d);
  last(&ev);
  EXPECT(ev.kind == NVTX_KIND_RANGE_POP && ev.domain_id == trtDomain,
         "domain pop");
  EXPECT(events() == 2, "case5 count");
  EXPECT(names() == 1, "no re-announce within the interval");

  // -------- Case 6: a second client copy shares domains --------
  // E.g. torch and the Python nvtx package in the same process.
  reset();
  static Client b;
  clientInit(&b);
  g_initClient = &b;
  EXPECT(initNvtx(getExportTable) == 1, "second client init");
  nvtxDomainHandle_t d2 =
      ((DomainCreateA_t)b.core2[NVTX_CBID_CORE2_DomainCreateA])("TensorRT-LLM");
  EXPECT(d2 == d, "same domain name should map to the same handle");
  EXPECT(names() == 0, "existing domain should not be re-announced on create");

  // -------- Case 7: resource naming fires nvtx_resource_name --------
  reset();
  ((NameOsThreadA_t)a.core[NVTX_CBID_CORE_NameOsThreadA])(1234, "worker");
  ((NameCudaStreamA_t)a.cudart[NVTX_CBID_CUDART_NameCudaStreamA])(
      (void *)(uintptr_t)0xabc, "compute_stream");
  EXPECT(names() == 2, "case7 names");
  EXPECT(events() == 0, "case7 should not capture events");

  // -------- Case 8: batch fires when full --------
  reset();
  // The default batch size from CMakeLists.txt is 256. Push that many
  // events on this thread; that should fire exactly one batch.
  for (int i = 0; i < 256; ++i)
    markA("x");
  EXPECT(events() == 256, "case8 events");
  EXPECT(batches() == 1, "case8 batches");
  // One more event — should sit in a new batch, not fire yet.
  markA("tail");
  EXPECT(events() == 257, "case8 tail event");
  EXPECT(batches() == 1, "case8 should still be 1 batch");

  // -------- Case 8b: a batch also fires once its oldest event is stale ------
  reset();
  usleep(150 * 1000); // > kMaxBatchAgeNs; "tail" above is now stale
  markA("late");
  EXPECT(batches() == 1, "stale batch should fire on the next event");

  // -------- Case 8c: long messages, truncation and the message arena -------
  // Messages up to 1023 bytes arrive intact; longer ones are cut there.
  static char longMsg[2001];
  memset(longMsg, 'x', 600);
  longMsg[600] = '\0';
  reset();
  markA(longMsg);
  msg = last(&ev);
  EXPECT(strlen(msg) == 600 && memcmp(msg, longMsg, 600) == 0,
         "600-byte message intact");
  memset(longMsg, 'y', 2000);
  longMsg[2000] = '\0';
  markA(longMsg);
  msg = last(&ev);
  EXPECT(strlen(msg) == 1023 && memcmp(msg, longMsg, 1023) == 0,
         "2000-byte message truncated to 1023");
  // The pending batch's messages share a 64 KB arena; a batch fires early
  // once it might not fit another 1024-byte message, well before 256 events.
  // Case 8b left the batch empty, so it now holds the 601 + 1024 bytes
  // above; at 601 bytes per message the first flush comes 105 events later
  // and the next one 108 after that.
  memset(longMsg, 'x', 600);
  longMsg[600] = '\0';
  reset();
  for (int i = 0; i < 105; ++i)
    markA(longMsg);
  EXPECT(batches() == 0, "arena not yet full");
  markA(longMsg);
  EXPECT(batches() == 1, "full arena fires the batch");
  for (int i = 0; i < 107; ++i)
    markA(longMsg);
  EXPECT(batches() == 1, "second arena not yet full");
  markA(longMsg);
  EXPECT(batches() == 2, "second full arena fires the batch");

  // -------- Case 9: force-capture off restores cold path --------
  force(0);
  reset();
  pushA("skipped");
  pop();
  // No USDT consumer attached, so without force-capture nothing should
  // run past the early exit.
  EXPECT(events() == 0, "case9 cold path");

  // -------- Case 10: cuInit points NVTX_INJECTION64_PATH at the library ----
  EXPECT(initCuda() == 1, "InitializeInjection returned non-1");
  Dl_info info;
  EXPECT(dladdr((void *)initNvtx, &info) && info.dli_fname, "dladdr");
  const char *inj = getenv("NVTX_INJECTION64_PATH");
  EXPECT(inj && strcmp(inj, info.dli_fname) == 0,
         "NVTX_INJECTION64_PATH should point at libparcagpucupti.so");

  fprintf(stderr, "PASS\n");
  return 0;
}
