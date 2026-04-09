// parcagpu — host-side CUDA profiler sources and shared BPF headers.
//
// This module exists primarily so external projects (e.g.
// opentelemetry-ebpf-profiler) can pull in the canonical BPF-side
// CUPTI definitions at ebpf/cupti_bpf.h via `go mod download`, the same
// way they consume github.com/parca-dev/usdt. The Go module has no Go
// source of its own — ebpf/ is a plain header directory.
module github.com/parca-dev/parcagpu

go 1.25.0
