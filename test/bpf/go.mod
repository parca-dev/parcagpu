module github.com/parca-dev/batch-kernel/test/bpf

go 1.25.1

require (
	github.com/cilium/ebpf v0.21.0
	github.com/gnurizen/sass-table v0.0.1
	go.opentelemetry.io/ebpf-profiler v0.0.0-00010101000000-000000000000
	golang.org/x/sys v0.41.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/klauspost/cpuid/v2 v2.2.10 // indirect
	github.com/minio/sha256-simd v1.0.1 // indirect
	github.com/rogpeppe/go-internal v1.14.1 // indirect
	go.opentelemetry.io/otel v1.41.0 // indirect
	go.opentelemetry.io/otel/metric v1.41.0 // indirect
)

replace go.opentelemetry.io/ebpf-profiler => ../../vendor/opentelemetry-ebpf-profiler
