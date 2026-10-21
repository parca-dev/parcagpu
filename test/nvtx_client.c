// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// Real NVTX v3 client for test-nvtx-real.sh. Uses the stock header-only
// nvtx3 implementation, so it finds libparcagpucupti.so only through
// NVTX_INJECTION64_PATH, exactly like torch / the Python nvtx package. It
// never initializes CUDA.
//
// Before the go file exists (no consumer attached yet) it creates a domain,
// registers a string and emits one mark, like libraries do at import time.
// The mark must not be captured. Once the go file appears it emits a fixed
// workload; test-nvtx-real.sh checks the BPF consumer saw exactly that.
//
// Usage: NVTX_INJECTION64_PATH=libparcagpucupti.so nvtx_client <go-file>

#define _GNU_SOURCE
#include <nvtx3/nvToolsExt.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define CORE_PUSHES 100
#define CORE_MARKS 10
#define CORE_RANGES 5
// One message that fits the collector's 1024-byte cap and one cut at it.
#define LONG_MARK_LEN 600
#define TRUNC_MARK_LEN 2000
#define EX_PUSHES 10
#define EX_COLOR 0xFF00FF00u
#define EX_PAYLOAD 7
// Well past PARCAGPU_NVTX_BATCH_SIZE, so several full batches fire.
#define DOMAIN_PUSHES 1000
#define DOMAIN_RANGES 3
// Spaced past the collector's 100ms max batch age.
#define DOMAIN_MARKS 5
#define DOMAIN_MARK_GAP_MS 150
#define WORKER_MARKS 50

static void sleep_ms(long ms) {
  struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

static nvtxEventAttributes_t attribs(void) {
  nvtxEventAttributes_t a;
  memset(&a, 0, sizeof(a));
  a.version = NVTX_VERSION;
  a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  return a;
}

static void *worker(void *arg) {
  (void)arg;
  nvtxNameOsThreadA((uint32_t)syscall(SYS_gettid), "nvtx-worker");
  for (int i = 0; i < WORKER_MARKS; i++)
    nvtxMarkA("worker_mark");
  // Thread exit flushes the batch via the collector's TLS destructor.
  return NULL;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <go-file>\n", argv[0]);
    return 2;
  }

  // Import-time setup, before any consumer is attached.
  nvtxDomainHandle_t dom = nvtxDomainCreateA("parcagpu-test");
  nvtxStringHandle_t step = nvtxDomainRegisterStringA(dom, "registered_step");
  nvtxMarkA("pre_attach");

  printf("client: waiting for %s\n", argv[1]);
  fflush(stdout);
  struct stat st;
  for (int i = 0; stat(argv[1], &st) != 0; i++) {
    if (i == 600) {
      fprintf(stderr, "client: timed out waiting for %s\n", argv[1]);
      return 1;
    }
    sleep_ms(100);
  }

  for (int i = 0; i < CORE_PUSHES; i++) {
    nvtxRangePushA("core_push");
    nvtxRangePop();
  }
  for (int i = 0; i < CORE_MARKS; i++)
    nvtxMarkA("core_mark");
  for (int i = 0; i < CORE_RANGES; i++)
    nvtxRangeEnd(nvtxRangeStartA("core_range"));

  static char long_mark[TRUNC_MARK_LEN + 1];
  memcpy(long_mark, "long:", 5);
  memset(long_mark + 5, 'x', LONG_MARK_LEN - 5);
  nvtxMarkA(long_mark);
  memcpy(long_mark, "trunc:", 6);
  memset(long_mark + 6, 'y', TRUNC_MARK_LEN - 6);
  nvtxMarkA(long_mark);

  nvtxEventAttributes_t ex = attribs();
  ex.colorType = NVTX_COLOR_ARGB;
  ex.color = EX_COLOR;
  ex.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  ex.payload.ullValue = EX_PAYLOAD;
  ex.messageType = NVTX_MESSAGE_TYPE_ASCII;
  ex.message.ascii = "ex_push";
  for (int i = 0; i < EX_PUSHES; i++) {
    nvtxRangePushEx(&ex);
    nvtxRangePop();
  }

  nvtxEventAttributes_t reg = attribs();
  reg.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  reg.message.registered = step;
  for (int i = 0; i < DOMAIN_PUSHES; i++) {
    nvtxDomainRangePushEx(dom, &reg);
    nvtxDomainRangePop(dom);
  }

  nvtxEventAttributes_t dr = attribs();
  dr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  dr.message.ascii = "domain_range";
  for (int i = 0; i < DOMAIN_RANGES; i++)
    nvtxDomainRangeEnd(dom, nvtxDomainRangeStartEx(dom, &dr));

  nvtxEventAttributes_t dm = attribs();
  dm.messageType = NVTX_MESSAGE_TYPE_ASCII;
  dm.message.ascii = "domain_mark";
  for (int i = 0; i < DOMAIN_MARKS; i++) {
    sleep_ms(DOMAIN_MARK_GAP_MS);
    nvtxDomainMarkEx(dom, &dm);
  }

  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  pthread_join(t, NULL);

  printf("client: done, expected events=%d\n",
         2 * CORE_PUSHES + CORE_MARKS + 2 * CORE_RANGES + 2 + 2 * EX_PUSHES +
             2 * DOMAIN_PUSHES + 2 * DOMAIN_RANGES + DOMAIN_MARKS +
             WORKER_MARKS);
  // The main thread's last batch fires from the injection's atexit flush.
  return 0;
}
