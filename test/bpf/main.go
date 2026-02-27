// Test program that loads the activity_parser BPF program, attaches it
// to the parcagpu:activity_batch USDT probe in the target shared library,
// and logs kernel activities received through the ring buffer.
//
// The USDT probe location and argument specs are parsed from the ELF
// .note.stapsdt section using pfelf, then populated into the BPF
// __bpf_usdt_specs map so bpf_usdt_arg() reads the correct registers.
//
// Usage:
//
//	go generate ./...
//	go build -o activity_parser .
//	sudo ./activity_parser -pid <PID> -lib <path/to/libparcagpucupti.so>
package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"go.opentelemetry.io/ebpf-profiler/libpf/pfelf"
	"golang.org/x/sys/unix"
)

//go:generate bpf2go -cc clang -cflags "-O2 -g -Wall -target bpf -D__TARGET_ARCH_x86 -D__x86_64__ -I../../vendor/opentelemetry-ebpf-profiler/support/ebpf" activityParser activity_parser.bpf.c

// KernelEvent matches struct kernel_event in the BPF program.
type KernelEvent struct {
	Start         uint64
	End           uint64
	CorrelationID uint32
	DeviceID      uint32
	StreamID      uint32
	GraphID       uint32
	GraphNodeID   uint64
	Name          [128]byte
}

const (
	statBatches    = 0
	statActivities = 1
	statKernels    = 2
	statDrops      = 3
)

func main() {
	pid := flag.Int("pid", 0, "PID of the target process")
	libPath := flag.String("lib", "", "Path to the shared library containing the USDT probe")
	verbose := flag.Bool("v", false, "Print every kernel event (default: summary only)")
	flag.Parse()

	if *pid == 0 || *libPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Resolve symlinks so uprobe attaches to the correct inode.
	realLib, err := filepath.EvalSymlinks(*libPath)
	if err != nil {
		log.Fatalf("Resolving symlinks for %s: %v", *libPath, err)
	}
	if realLib != *libPath {
		log.Printf("Resolved %s -> %s", *libPath, realLib)
	}

	// Raise memlock rlimit for BPF maps.
	if err := raiseMemlock(); err != nil {
		log.Printf("Warning: failed to raise memlock rlimit: %v", err)
	}

	// Load pre-compiled BPF objects.
	objs := activityParserObjects{}
	if err := loadActivityParserObjects(&objs, nil); err != nil {
		var ve *ebpf.VerifierError
		if errors.As(err, &ve) {
			log.Fatalf("Verifier error loading BPF objects:\n%+v", ve)
		}
		log.Fatalf("Loading BPF objects: %v", err)
	}
	defer objs.Close()

	// Parse USDT probes from the shared library's .note.stapsdt section.
	ef, err := pfelf.Open(realLib)
	if err != nil {
		log.Fatalf("Opening ELF %s: %v", realLib, err)
	}
	defer ef.Close()

	if err := ef.LoadSections(); err != nil {
		log.Fatalf("Loading ELF sections: %v", err)
	}

	probes, err := ef.ParseUSDTProbes()
	if err != nil {
		log.Fatalf("Parsing USDT probes: %v", err)
	}

	// Find parcagpu:activity_batch probe(s) and attach uprobe at each site.
	ex, err := link.OpenExecutable(realLib)
	if err != nil {
		log.Fatalf("Opening executable %s: %v", realLib, err)
	}

	var links []link.Link
	var specID uint32
	for _, probe := range probes {
		if probe.Provider != "parcagpu" || probe.Name != "activity_batch" {
			continue
		}

		// Parse the stapsdt argument spec into a bpf_usdt_spec.
		spec, err := pfelf.ParseUSDTArguments(probe.Arguments)
		if err != nil {
			log.Fatalf("Parsing USDT args %q: %v", probe.Arguments, err)
		}

		// Store spec in the BPF map so bpf_usdt_arg() can look it up.
		specBytes := pfelf.USDTSpecToBytes(spec)
		if err := objs.BpfUsdtSpecs.Put(specID, specBytes); err != nil {
			log.Fatalf("Populating USDT spec map: %v", err)
		}

		// Cookie: spec_id in high 32 bits (bpf_usdt_arg reads it via bpf_get_attach_cookie).
		cookie := uint64(specID) << 32

		log.Printf("USDT probe parcagpu:activity_batch at offset 0x%x, args=%q, spec_id=%d",
			probe.Location, probe.Arguments, specID)

		up, err := ex.Uprobe("activity_batch", objs.HandleActivityBatch, &link.UprobeOptions{
			Address:      probe.Location,
			PID:          *pid,
			Cookie:       cookie,
			RefCtrOffset: probe.SemaphoreOffset,
		})
		if err != nil {
			log.Fatalf("Attaching uprobe at offset 0x%x: %v", probe.Location, err)
		}
		links = append(links, up)
		specID++
	}

	if len(links) == 0 {
		log.Fatalf("No parcagpu:activity_batch USDT probes found in %s", realLib)
	}
	defer func() {
		for _, l := range links {
			l.Close()
		}
	}()

	// Open ring buffer reader.
	rd, err := ringbuf.NewReader(objs.Events)
	if err != nil {
		log.Fatalf("Opening ring buffer: %v", err)
	}
	defer rd.Close()

	// Handle signals.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	// Also watch for the target process to exit.
	done := make(chan struct{})
	go func() {
		for {
			if err := syscall.Kill(*pid, 0); err != nil {
				log.Printf("Target process %d exited", *pid)
				close(done)
				return
			}
			time.Sleep(500 * time.Millisecond)
		}
	}()

	go func() {
		select {
		case <-sig:
		case <-done:
		}
		rd.Close()
	}()

	log.Printf("Attached %d USDT probe(s) in %s (PID %d)", len(links), realLib, *pid)
	log.Printf("Waiting for kernel activity events...")

	var eventCount uint64
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	go func() {
		for range ticker.C {
			printStats(&objs, eventCount)
		}
	}()

	var event KernelEvent
	for {
		record, err := rd.Read()
		if err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				break
			}
			log.Printf("Reading from ring buffer: %v", err)
			continue
		}

		if err := binary.Read(bytes.NewBuffer(record.RawSample), binary.LittleEndian, &event); err != nil {
			log.Printf("Parsing event: %v", err)
			continue
		}

		eventCount++
		if *verbose {
			name := cString(event.Name[:])
			duration := event.End - event.Start
			fmt.Printf("kernel: name=%-40s corr=%-6d dev=%d stream=%d graph=%-3d duration=%dns\n",
				name, event.CorrelationID, event.DeviceID, event.StreamID, event.GraphID, duration)
		}
	}

	fmt.Println()
	log.Printf("Final stats:")
	printStats(&objs, eventCount)
}

func printStats(objs *activityParserObjects, eventCount uint64) {
	var batches, activities, kernels, drops uint64
	batchKey := uint32(statBatches)
	activityKey := uint32(statActivities)
	kernelKey := uint32(statKernels)
	dropsKey := uint32(statDrops)

	objs.Stats.Lookup(&batchKey, &batches)
	objs.Stats.Lookup(&activityKey, &activities)
	objs.Stats.Lookup(&kernelKey, &kernels)
	objs.Stats.Lookup(&dropsKey, &drops)

	log.Printf("  batches=%d activities_scanned=%d kernels_found=%d events_received=%d drops=%d",
		batches, activities, kernels, eventCount, drops)
}

func raiseMemlock() error {
	return unix.Setrlimit(unix.RLIMIT_MEMLOCK, &unix.Rlimit{
		Cur: unix.RLIM_INFINITY,
		Max: unix.RLIM_INFINITY,
	})
}

func cString(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		return string(b[:i])
	}
	return string(b)
}

// Ensure KernelEvent matches the BPF struct size.
var _ = [1]struct{}{}[unsafe.Sizeof(KernelEvent{})-168]
