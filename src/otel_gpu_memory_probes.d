/* GPU allocation probes using the otel_memory contract from
 * opentelemetry-ebpf-profiler PR #1747, under a separate provider so GPU
 * memory never merges into a host heap profile.
 *
 *   alloc(ptr, size, weighted_bytes)   sampled allocations only
 *   free(ptr)                          every free; consumers drop
 *                                      pointers they never saw allocated
 *
 * Consumers read the arguments from the ABI argument registers (rdi/rsi/rdx
 * on x86-64, x0/x1/x2 on arm64), not from the SDT argument descriptors.
 * gpu_memory.cpp fires these from noinline functions with the operands
 * pinned to those registers, and the build checks the notes (see
 * scripts/check-otel-usdt-regs.sh).
 */
provider otel_gpu_memory {
  probe alloc(void *ptr, uint64_t size, uint64_t weighted_bytes);
  probe free(void *ptr);
};
