// Compatibility shim for including the otel-ebpf-profiler usdt.h and
// usdt_args.h from our vmlinux.h-based BPF program.
//
// Must be included AFTER vmlinux.h + bpf_helpers.h but BEFORE usdt_args.h.
// Defines the include guards for bpfdefs.h and types.h so those vendor
// headers (and their deep dependency trees) are skipped entirely.

#ifndef USDT_COMPAT_H
#define USDT_COMPAT_H

// Skip vendor bpfdefs.h and types.h — we already have everything from
// vmlinux.h.
#define OPTI_BPFDEFS_H
#define OPTI_TYPES_H

// Macros expected by usdt_args.h (normally from bpfdefs.h).
#define EBPF_INLINE __always_inline
#define UNUSED __attribute__((unused))

// bpf_map_def — usdt_args.h uses "extern bpf_map_def __bpf_usdt_specs"
// (C++ style, no struct keyword), so we need a typedef.
typedef struct bpf_map_def {
  unsigned int type;
  unsigned int key_size;
  unsigned int value_size;
  unsigned int max_entries;
  unsigned int map_flags;
} bpf_map_def;

// bpf2go passes -target bpfel (not the real platform triple), so the
// usual compiler builtins (__x86_64__, __aarch64__) are never set.
// Bridge from bpf2go's __TARGET_ARCH_* defines to the builtins that
// usdt_args.h checks for pt_regs layout.
#if defined(__TARGET_ARCH_x86)
#define __x86_64__
#elif defined(__TARGET_ARCH_arm64)
#define __aarch64__
#endif

#endif // USDT_COMPAT_H
