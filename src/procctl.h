/*
 * procctl.h — process side of procctl (SPEC.md).
 *
 * Vendored from github.com/parca-dev/procctl v0.1.0-alpha.1 (c/procctl.h).
 *
 * A node agent pushes configuration to this process through a shared-memory
 * region the process creates and maps read-only; the process answers through
 * a small state block in its own memory, which the agent reads with
 * process_vm_readv. The process never writes shared memory.
 *
 * Single-header library. In exactly one translation unit:
 *
 *     #define PROCCTL_IMPLEMENTATION
 *     #include "procctl.h"
 *
 * That translation unit needs _GNU_SOURCE (or -std=gnu11, or C++).
 * No allocation, no threads, no constructors, no signal or fork handlers.
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright The procctl Authors.
 */
#ifndef PROCCTL_H
#define PROCCTL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PROCCTL_VERSION 1 /* wire format; part of the memfd name */

#ifndef PROCCTL_DEF
#define PROCCTL_DEF __attribute__((visibility("hidden")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Wire structs (SPEC.md): little-endian, no implicit padding. The Go agent's
 * types are generated from these with cgo -godefs. */

struct procctl_header { /* region offset 0; written once, before sealing */
  uint8_t magic[8];     /* "PROC_CTL" */
  uint16_t version;     /* PROCCTL_VERSION */
  uint16_t header_size;
  uint32_t page_size;   /* the agent area starts here */
  uint64_t total_size;  /* of the memfd */
  uint8_t consumer[32]; /* NUL-padded [a-z0-9_]{1,31}; also in the memfd name */
  uint32_t schema_version;
  uint32_t state_size;  /* capacity of the state block */
  uint64_t state_addr;  /* the state block, in this process's address space */
  uint64_t instance_id; /* random; the state block repeats it */
};

struct procctl_ctrl {    /* region offset page_size; written by the owner agent */
  uint64_t seq;          /* seqlock: odd while writing; an even value is a generation */
  uint64_t owner_id;     /* 0 = no owner */
  uint64_t heartbeat_ns; /* the owner's clock; changes while it is alive */
  uint64_t lease_ns;     /* no heartbeat change for this long = no owner */
  uint32_t payload_len;  /* payload bytes that follow; 0 = use defaults */
  uint32_t payload_crc;  /* CRC32C of the payload */
  uint32_t flags;        /* PROCCTL_RELEASED */
  uint32_t reserved;
};

struct procctl_state {  /* in this process's memory, at header.state_addr */
  uint8_t magic[8];     /* "PROCSTAT" */
  uint64_t seq;         /* seqlock */
  uint64_t instance_id; /* = header.instance_id */
  uint64_t applied_gen; /* latest generation applied (acked) */
  uint32_t pid;         /* getpid() of the writer: a forked child's copy is stale */
  uint32_t len;         /* consumer state bytes that follow */
};

#define PROCCTL_RELEASED 1u /* procctl_ctrl.flags: the owner left; a successor may claim now */

/* Bytes of state_storage for state_capacity bytes of consumer state. */
#define PROCCTL_STATE_SIZE(cap) (((sizeof(struct procctl_state) + (size_t)(cap)) + 7u) & ~(size_t)7u)
#define PROCCTL_MAX_STATE_SIZE 65536u

#ifdef __cplusplus
#define PROCCTL__STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define PROCCTL__STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif
PROCCTL__STATIC_ASSERT(sizeof(struct procctl_header) == 80, "header size");
PROCCTL__STATIC_ASSERT(sizeof(struct procctl_ctrl) == 48, "control block size");
PROCCTL__STATIC_ASSERT(sizeof(struct procctl_state) == 40, "state block size");

typedef struct procctl {
  uint64_t opaque[24];
} procctl_t;

typedef struct procctl_config {
  const char *consumer;      /* [a-z0-9_]{1,31} */
  uint32_t schema_version;   /* yours: tells the agent how to read your payload and state */
  uint32_t payload_capacity; /* largest payload the agent may send */
  uint32_t state_capacity;   /* largest state you will report */
  void *state_storage;       /* PROCCTL_STATE_SIZE(state_capacity) bytes, 8-aligned, static */
} procctl_config_t;

/* Creates and maps the region. 0 or -errno; on error, run on your defaults. */
PROCCTL_DEF int procctl_init(procctl_t *ctx, const procctl_config_t *cfg);

/* Checks for a new configuration; cheap enough for a hot path (one atomic load
 * when nothing changed). Returns 0 = no change; > 0 = payload bytes copied to
 * buf, *gen set; -ENODATA = new generation without payload: use your
 * defaults, *gen set; -EAGAIN = mid-write, try again later; -E2BIG = buf too
 * small (the generation is skipped); -ENODEV = not initialized. Each
 * generation is returned to exactly one caller. */
static inline int procctl_poll(procctl_t *ctx, void *buf, size_t cap, uint64_t *gen);

/* Reports that generation gen is applied. 0, -ECHILD in a forked child, or
 * -ENODEV. */
PROCCTL_DEF int procctl_ack(procctl_t *ctx, uint64_t gen);

/* Publishes your consumer state (up to state_capacity bytes). Rate-limit it:
 * the agent reads it with a seqlock. 0, -E2BIG, -ECHILD or -ENODEV. */
PROCCTL_DEF int procctl_state_set(procctl_t *ctx, const void *state, size_t len);

/* True when no agent owns the region, or its heartbeat stopped for longer than
 * its lease: revert to your defaults. Call it periodically (now_ns 0 = now).
 * When the lease recovers, the next poll returns the current configuration
 * again. */
PROCCTL_DEF bool procctl_lease_expired(procctl_t *ctx, uint64_t now_ns);

/* In a forked child that wants its own channel: drops the inherited region and
 * creates a new one. 0 or -errno. */
PROCCTL_DEF int procctl_reinit(procctl_t *ctx);
PROCCTL_DEF void procctl_close(procctl_t *ctx);

/* CLOCK_BOOTTIME in nanoseconds. */
PROCCTL_DEF uint64_t procctl_now_ns(void);

/* ---- internals (shared with the inline poll) ------------------------------ */

struct procctl__ctx {
  const uint64_t *seq;   /* &ctrl->seq in the read-only mapping */
  uint64_t last_seen;    /* atomic: generation last returned by poll */
  const uint8_t *map;
  size_t map_size;
  struct procctl_state *state;
  uint64_t hb_seen;      /* atomic: heartbeat last seen... */
  uint64_t hb_local_ns;  /* atomic: ...and our clock when it changed */
  uint32_t state_capacity;
  uint32_t payload_capacity;
  uint32_t schema_version;
  int32_t fd;
  int32_t pid;           /* pid that created the region */
  int32_t lock;          /* atomic: serializes state writes */
  int32_t expired;       /* atomic: last lease result */
  int32_t ready;
  char consumer[32];
};

PROCCTL_DEF int procctl__poll_slow(struct procctl__ctx *c, uint64_t s, void *buf, size_t cap,
                                   uint64_t *gen);

static inline int procctl_poll(procctl_t *ctx, void *buf, size_t cap, uint64_t *gen) {
  struct procctl__ctx *c = (struct procctl__ctx *)(void *)ctx->opaque;
  if (__builtin_expect(!c->ready, 0)) return -19; /* -ENODEV */
  uint64_t s = __atomic_load_n(c->seq, __ATOMIC_ACQUIRE);
  if (__builtin_expect(s == __atomic_load_n(&c->last_seen, __ATOMIC_RELAXED), 1)) return 0;
  return procctl__poll_slow(c, s, buf, cap, gen);
}

#ifdef __cplusplus
}
#endif
#endif /* PROCCTL_H */

/* ========================================================================== */
#ifdef PROCCTL_IMPLEMENTATION
#ifndef PROCCTL_IMPLEMENTATION_DONE
#define PROCCTL_IMPLEMENTATION_DONE

#if !defined(__cplusplus) && defined(__STRICT_ANSI__) && !defined(_GNU_SOURCE) &&             \
    !defined(_DEFAULT_SOURCE)
#error "procctl implementation needs _GNU_SOURCE (or -std=gnu11): define it before any #include"
#endif

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001u
#define MFD_ALLOW_SEALING 0x0002u
#endif
#ifndef MFD_NOEXEC_SEAL
#define MFD_NOEXEC_SEAL 0x0008u
#endif
#ifndef F_ADD_SEALS
#define F_ADD_SEALS 1033
#define F_SEAL_SEAL 0x0001
#define F_SEAL_SHRINK 0x0002
#define F_SEAL_GROW 0x0004
#endif

PROCCTL__STATIC_ASSERT(sizeof(struct procctl__ctx) <= sizeof(procctl_t), "procctl_t too small");

#define PC__LOAD(x) __atomic_load_n(&(x), __ATOMIC_ACQUIRE)

static struct procctl__ctx *pc__c(procctl_t *ctx) { return (struct procctl__ctx *)(void *)ctx->opaque; }

static size_t pc__page(void) { return (size_t)sysconf(_SC_PAGESIZE); }

static const struct procctl_ctrl *pc__ctrl(const struct procctl__ctx *c) {
  return (const struct procctl_ctrl *)(const void *)(c->map + pc__page());
}

/* CRC32C (Castagnoli), reflected, four bits at a time: small, and only run
 * when the configuration changes. */
static uint32_t pc__crc32c(const uint8_t *p, size_t n) {
  static const uint32_t t[16] = {
      0x00000000u, 0x105ec76fu, 0x20bd8edeu, 0x30e349b1u, 0x417b1dbcu, 0x5125dad3u,
      0x61c69362u, 0x7198540du, 0x82f63b78u, 0x92a8fc17u, 0xa24bb5a6u, 0xb21572c9u,
      0xc38d26c4u, 0xd3d3e1abu, 0xe330a81au, 0xf36e6f75u};
  uint32_t crc = ~0u;
  while (n--) {
    crc ^= *p++;
    crc = (crc >> 4) ^ t[crc & 15u];
    crc = (crc >> 4) ^ t[crc & 15u];
  }
  return ~crc;
}

PROCCTL_DEF uint64_t procctl_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t pc__random64(void) {
  uint64_t v = 0;
  if (syscall(SYS_getrandom, &v, sizeof v, 1 /* GRND_NONBLOCK */) != (long)sizeof v)
    v = procctl_now_ns() ^ ((uint64_t)getpid() << 32);
  return v ? v : 1;
}

static int pc__valid_consumer(const char *s) {
  size_t n = 0;
  if (!s) return 0;
  for (; s[n]; n++) {
    char ch = s[n];
    if (n >= 31 || !((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_')) return 0;
  }
  return n > 0;
}

/* ---- state block (seqlock in our own memory) ----------------------------- */

static int pc__state_begin(struct procctl__ctx *c) {
  if (getpid() != c->pid) return -ECHILD; /* never write the parent's block */
  for (unsigned spins = 0;; spins++) {
    int exp = 0;
    if (__atomic_compare_exchange_n(&c->lock, &exp, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) break;
    if (spins > 64) sched_yield();
  }
  __atomic_store_n(&c->state->seq, c->state->seq + 1, __ATOMIC_RELAXED); /* odd */
  __atomic_thread_fence(__ATOMIC_RELEASE);
  return 0;
}

static void pc__state_end(struct procctl__ctx *c) {
  __atomic_store_n(&c->state->seq, c->state->seq + 1, __ATOMIC_RELEASE); /* even */
  __atomic_store_n(&c->lock, 0, __ATOMIC_RELEASE);
}

PROCCTL_DEF int procctl_ack(procctl_t *ctx, uint64_t gen) {
  struct procctl__ctx *c = pc__c(ctx);
  if (!c->ready) return -ENODEV;
  int r = pc__state_begin(c);
  if (r) return r;
  if (gen > c->state->applied_gen) c->state->applied_gen = gen;
  pc__state_end(c);
  return 0;
}

PROCCTL_DEF int procctl_state_set(procctl_t *ctx, const void *state, size_t len) {
  struct procctl__ctx *c = pc__c(ctx);
  if (!c->ready) return -ENODEV;
  if (len > c->state_capacity) return -E2BIG;
  int r = pc__state_begin(c);
  if (r) return r;
  memcpy((uint8_t *)c->state + sizeof *c->state, state, len);
  c->state->len = (uint32_t)len;
  pc__state_end(c);
  return 0;
}

/* ---- region ----------------------------------------------------------------- */

static int pc__create(struct procctl__ctx *c) {
  size_t P = pc__page();
  size_t area = (sizeof(struct procctl_ctrl) + c->payload_capacity + P - 1) / P * P;
  size_t total = P + area;
  char name[48];
  snprintf(name, sizeof name, "procctl:%s-v%d", c->consumer, PROCCTL_VERSION);

  int fd = (int)syscall(SYS_memfd_create, name, MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_NOEXEC_SEAL);
  if (fd < 0 && errno == EINVAL) /* kernels before 6.3 */
    fd = (int)syscall(SYS_memfd_create, name, MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (fd < 0) return -errno;

  struct procctl_header h;
  memset(&h, 0, sizeof h);
  memcpy(h.magic, "PROC_CTL", 8);
  h.version = PROCCTL_VERSION;
  h.header_size = (uint16_t)sizeof h;
  h.page_size = (uint32_t)P;
  h.total_size = total;
  memcpy(h.consumer, c->consumer, strlen(c->consumer));
  h.schema_version = c->schema_version;
  h.state_size = (uint32_t)PROCCTL_STATE_SIZE(c->state_capacity);
  h.state_addr = (uint64_t)(uintptr_t)c->state;
  h.instance_id = pc__random64();

  /* The state block first, so it is valid by the time an agent can find the
   * region; the seals last, so a sealed region has a complete header. */
  memset(c->state, 0, h.state_size);
  memcpy(c->state->magic, "PROCSTAT", 8);
  c->state->instance_id = h.instance_id;
  c->state->pid = (uint32_t)c->pid;
  c->state->seq = 2;

  void *m = MAP_FAILED;
  if (ftruncate(fd, (off_t)total) != 0 || pwrite(fd, &h, sizeof h, 0) != (ssize_t)sizeof h ||
      fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0 ||
      (m = mmap(NULL, total, PROT_READ, MAP_SHARED, fd, 0)) == MAP_FAILED) {
    int e = errno ? errno : EIO;
    close(fd);
    return -e;
  }
  c->fd = fd;
  c->map = (const uint8_t *)m;
  c->map_size = total;
  c->seq = &pc__ctrl(c)->seq;
  __atomic_store_n(&c->last_seen, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&c->hb_seen, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&c->hb_local_ns, procctl_now_ns(), __ATOMIC_RELAXED);
  __atomic_store_n(&c->expired, 1, __ATOMIC_RELAXED);
  c->ready = 1;
  return 0;
}

PROCCTL_DEF int procctl_init(procctl_t *ctx, const procctl_config_t *cfg) {
  struct procctl__ctx *c = pc__c(ctx);
  memset(ctx, 0, sizeof *ctx);
  if (!cfg || !pc__valid_consumer(cfg->consumer)) return -EINVAL;
  if (PROCCTL_STATE_SIZE(cfg->state_capacity) > PROCCTL_MAX_STATE_SIZE) return -E2BIG;
  if (!cfg->state_storage || ((uintptr_t)cfg->state_storage & 7u)) return -EINVAL;
  c->state = (struct procctl_state *)cfg->state_storage;
  c->state_capacity = cfg->state_capacity;
  c->payload_capacity = cfg->payload_capacity;
  c->schema_version = cfg->schema_version;
  c->pid = getpid();
  c->fd = -1;
  memcpy(c->consumer, cfg->consumer, strlen(cfg->consumer));
  return pc__create(c);
}

PROCCTL_DEF void procctl_close(procctl_t *ctx) {
  struct procctl__ctx *c = pc__c(ctx);
  if (!c->ready) return;
  c->ready = 0;
  munmap((void *)c->map, c->map_size);
  close(c->fd);
}

PROCCTL_DEF int procctl_reinit(procctl_t *ctx) {
  struct procctl__ctx *c = pc__c(ctx);
  if (!c->ready) return -ENODEV;
  procctl_close(ctx);
  c->pid = getpid();
  __atomic_store_n(&c->lock, 0, __ATOMIC_RELAXED); /* a thread of the parent may have held it */
  return pc__create(c);
}

/* ---- configuration (seqlock reader) ---------------------------------------- */

PROCCTL_DEF int procctl__poll_slow(struct procctl__ctx *c, uint64_t s, void *buf, size_t cap,
                                   uint64_t *gen) {
  const struct procctl_ctrl *cb = pc__ctrl(c);
  uint64_t old = __atomic_load_n(&c->last_seen, __ATOMIC_RELAXED);
  if (s == old) return 0;
  if ((s & 1u) || s == 0) return -EAGAIN;
  uint32_t n = PC__LOAD(cb->payload_len), crc = PC__LOAD(cb->payload_crc);
  if (sizeof *cb + (size_t)n > c->map_size - pc__page()) return -EAGAIN; /* torn */
  /* Claim the generation first, so exactly one caller gets it. */
  if (!__atomic_compare_exchange_n(&c->last_seen, &old, s, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
    return 0;
  if (n > cap) return -E2BIG;
  /* The agent writes the payload in 8-byte words; read it the same way. */
  const uint64_t *src = (const uint64_t *)(const void *)((const uint8_t *)cb + sizeof *cb);
  for (size_t off = 0; off < n; off += 8) {
    uint64_t w = __atomic_load_n(&src[off / 8], __ATOMIC_RELAXED);
    memcpy((uint8_t *)buf + off, &w, n - off < 8 ? n - off : 8);
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  if (PC__LOAD(cb->seq) != s || pc__crc32c((const uint8_t *)buf, n) != crc) {
    /* Torn: hand the generation back so the next poll retries it. */
    __atomic_compare_exchange_n(&c->last_seen, &s, old, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return -EAGAIN;
  }
  if (gen) *gen = s;
  return n ? (int)n : -ENODATA;
}

/* ---- lease ------------------------------------------------------------------- */

PROCCTL_DEF bool procctl_lease_expired(procctl_t *ctx, uint64_t now_ns) {
  struct procctl__ctx *c = pc__c(ctx);
  if (!c->ready) return true;
  if (!now_ns) now_ns = procctl_now_ns();
  const struct procctl_ctrl *cb = pc__ctrl(c);
  uint64_t owner = PC__LOAD(cb->owner_id), lease = PC__LOAD(cb->lease_ns);
  uint64_t hb = PC__LOAD(cb->heartbeat_ns);
  if (hb != __atomic_load_n(&c->hb_seen, __ATOMIC_RELAXED)) {
    __atomic_store_n(&c->hb_seen, hb, __ATOMIC_RELAXED);
    __atomic_store_n(&c->hb_local_ns, now_ns, __ATOMIC_RELAXED);
  }
  bool expired = owner == 0 ||
                 (lease && now_ns - __atomic_load_n(&c->hb_local_ns, __ATOMIC_RELAXED) > lease);
  if (__atomic_exchange_n(&c->expired, expired, __ATOMIC_RELAXED) && !expired)
    __atomic_store_n(&c->last_seen, 0, __ATOMIC_RELAXED); /* recovered: deliver the current config again */
  return expired;
}

#endif /* PROCCTL_IMPLEMENTATION_DONE */
#endif /* PROCCTL_IMPLEMENTATION */
