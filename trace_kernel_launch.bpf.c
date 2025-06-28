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

SEC("uprobe/trace_kernel_launch")
int trace_kernel_launch(struct pt_regs *ctx) {
    bpf_printk("trace_kernel_launch fired!\n");
    
    // We're now attaching to launchKernelTiming(id: u32, duration_bits: u32)
    // Parameters: RDI = id (u32), RSI = duration_bits (u32)
    
    __u32 kernel_id = (__u32)PT_REGS_PARM1(ctx);        // RDI - first parameter (id)
    __u32 duration_bits = (__u32)PT_REGS_PARM2(ctx);    // RSI - second parameter (duration_bits)
    
    bpf_printk("launchKernelTiming: kernel_id=%u, duration_bits=0x%x\n", 
               kernel_id, duration_bits);
    
    // Send the actual timing data from the function parameters
    struct kernel_timing timing = {
        .kernel_id = kernel_id,
        .duration_bits = duration_bits,
    };
    
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &timing, sizeof(timing));
    
    return 0;
}

char _license[] SEC("license") = "GPL";