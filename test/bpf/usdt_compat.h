// Compatibility shim for including parca-dev/usdt's usdt_args.h from our
// vmlinux.h-based BPF program.
//
// Must be included AFTER vmlinux.h + bpf_helpers.h but BEFORE usdt_args.h.
// Provides macros that usdt_args.h expects (normally from kernel.h) and
// bridges bpf2go architecture defines to standard compiler builtins.

#ifndef USDT_COMPAT_H
#define USDT_COMPAT_H

// Macros expected by usdt_args.h (normally from kernel.h).
#define EBPF_INLINE __always_inline
#define UNUSED __attribute__((unused))

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
