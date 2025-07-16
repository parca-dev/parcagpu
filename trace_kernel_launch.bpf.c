#include <linux/ptrace.h>
#include <linux/types.h>
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct kernel_timing {
    __u32 kernel_id;
    __u32 duration_bits; // float32 as raw bits
};

struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
    __uint(max_entries, 1024);
} events SEC(".maps");

// USDT probe for parcagpu:kernel_launch
// The probe!(parcagpu, kernel_launch, id, ms) in Rust code passes two arguments:
// - arg0: id (u32)
// - arg1: ms (f32) - but we'll receive it as u32 bits
//
// When attached as a uprobe to a USDT probe, the arguments are passed in registers
SEC("uprobe/trace_kernel_launch")
int trace_kernel_launch(struct pt_regs *ctx) {
    bpf_printk("trace_kernel_launch fired!\n");

    // USDT probes on x86_64 typically pass arguments via registers
    // The exact registers depend on the calling convention and USDT implementation
    // For SystemTap-style USDT probes, arguments are often in:
    // arg1: RDI or RSI
    // arg2: RSI or RDX

    // Try the standard calling convention first
    __u32 kernel_id = (__u32)PT_REGS_PARM1(ctx);
    __u32 duration_bits = (__u32)PT_REGS_PARM2(ctx);

    bpf_printk("USDT kernel_launch: kernel_id=%u, duration_bits=0x%x\n",
               kernel_id, duration_bits);

    // Send the timing data through perf event
    struct kernel_timing timing = {
        .kernel_id = kernel_id,
        .duration_bits = duration_bits,
    };

    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &timing, sizeof(timing));

    return 0;
}

char _license[] SEC("license") = "GPL";