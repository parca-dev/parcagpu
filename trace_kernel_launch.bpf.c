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

SEC("uprobe/parcagpu_kernel_launch")
int trace_kernel_launch(struct pt_regs *ctx) {
    struct kernel_timing timing = {};

    // Read USDT probe arguments from specific registers
    // Based on readelf output: Arguments: -8@%rdx -8@%rax
    // arg0: kernel_id (u32) - in rdx register
    // arg1: duration_ms (f32) - in rax register
    timing.kernel_id = (__u32)ctx->rdx;        // Read from RDX register
    timing.duration_bits = (__u32)ctx->rax;    // Read from RAX register

    // Submit event to userspace via perf event array
    if (bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &timing, sizeof(timing))) {
        // If output fails, we can log an error or handle it as needed
        bpf_printk("Failed to output kernel timing event: kernel_id=%u duration_bits=%u\n",
                   timing.kernel_id, timing.duration_bits);
    } else {
        // Optionally log successful event submission
        bpf_printk("Kernel launch traced: kernel_id=%u duration_bits=%u\n",
                   timing.kernel_id, timing.duration_bits);
    }

    return 0;
}

char _license[] SEC("license") = "GPL";