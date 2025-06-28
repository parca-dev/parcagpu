package main

import (
	"context"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
	"usdt_tracer/util"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/perf"
	"github.com/cilium/ebpf/rlimit"
)

// KernelTiming matches the struct from the eBPF program
type KernelTiming struct {
	KernelID     uint32
	DurationBits uint32 // float32 as raw bits
}

// USDTProbe represents a USDT probe found in ELF
type USDTProbe struct {
	Provider  string
	Name      string
	Location  uint64
	Base      uint64
	Semaphore uint64
	Arguments string
}

const bpfObjectPath = "target/bpf/trace_kernel_launch.bpf.o"

func main() {
	if len(os.Args) < 2 {
		fmt.Println("eBPF USDT Tracer for parcagpu kernel_launch probe")
		fmt.Println("Usage: ./usdt_tracer <target_pid>")
		fmt.Println("Example: ./usdt_tracer 1234")
		os.Exit(1)
	}

	targetPID := os.Args[1]
	pid, err := strconv.Atoi(targetPID)
	if err != nil {
		log.Fatalf("Invalid PID: %s", err)
	}

	// Remove memory limit for eBPF
	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatalf("Failed to remove memlock limit: %v", err)
	}

	fmt.Printf("Loading eBPF program: %s\n", bpfObjectPath)

	// Load the compiled eBPF program
	spec, err := ebpf.LoadCollectionSpec(bpfObjectPath)
	if err != nil {
		log.Fatalf("Failed to load eBPF spec: %v", err)
	}

	coll, err := ebpf.NewCollection(spec)
	if err != nil {
		log.Fatalf("Failed to create eBPF collection: %v", err)
	}
	defer coll.Close()

	// Get the eBPF program for attaching
	prog := coll.Programs["trace_kernel_launch"]
	if prog == nil {
		log.Fatal("trace_kernel_launch program not found in eBPF collection")
	}

	fmt.Printf("Attaching to USDT probe parcagpu:kernel_launch in PID %d\n", pid)

	// Find the target executable path from PID
	execPath := fmt.Sprintf("/proc/%d/exe", pid)
	realPath, err := os.Readlink(execPath)
	if err != nil {
		log.Fatalf("Failed to resolve executable path for PID %d: %v", pid, err)
	}

	fmt.Printf("Target executable: %s\n", realPath)

	// Try to find the parcagpu shared library in the process memory maps
	libPath := "./target/release/libparcagpu.so"

	// Read /proc/PID/maps to find the actual loaded library path
	mapsPath := fmt.Sprintf("/proc/%d/maps", pid)
	mapsData, err := os.ReadFile(mapsPath)
	if err == nil {
		mapsContent := string(mapsData)
		// Look for libparcagpu.so in the memory maps
		for _, line := range strings.Split(mapsContent, "\n") {
			if strings.Contains(line, "libparcagpu.so") {
				fields := strings.Fields(line)
				if len(fields) >= 6 {
					libPath = fields[5]
					break
				}
			}
		}
	}

	fmt.Printf("Attaching to library: %s\n", libPath)

	// Find the library's base address in the process memory
	baseAddr, err := findLibraryBaseAddress(pid, libPath)
	if err != nil {
		log.Fatalf("Failed to find library base address: %v", err)
	}
	fmt.Printf("Library loaded at base address: 0x%x\n", baseAddr)

	// Open the library executable
	ex, err := link.OpenExecutable(libPath)
	if err != nil {
		log.Fatalf("Failed to open executable/library: %v", err)
	}

	// Parse USDT probes from ELF file
	probes, err := parseUSDTProbes(libPath)
	if err != nil {
		log.Fatalf("Failed to parse USDT probes: %v", err)
	}

	// Find all parcagpu:kernel_launch probes
	var targetProbes []USDTProbe
	for _, probe := range probes {
		if probe.Provider == "parcagpu" && probe.Name == "kernel_launch" {
			targetProbes = append(targetProbes, probe)
		}
	}

	if len(targetProbes) == 0 {
		log.Fatal("USDT probe parcagpu:kernel_launch not found in library")
	}

	fmt.Printf("Found %d USDT probe locations for parcagpu:kernel_launch\n", len(targetProbes))

	// Attach uprobe to all USDT probe locations
	var links []link.Link
	for _, probe := range targetProbes {
		// Attach to the launchKernelTiming function that receives id and duration
		fmt.Printf("Attaching to launchKernelTiming function\n")

		l, err := ex.Uprobe("launchKernelTiming", prog, nil)
		if err != nil {
			log.Fatalf("Failed to attach to USDT probe at offset 0x%x: %v", probe.Location, err)
		}
		links = append(links, l)
		fmt.Printf("Successfully attached to USDT probe at location 0x%x\n", probe.Location)
	}

	// Defer closing all links
	defer func() {
		for _, l := range links {
			l.Close()
		}
	}()

	fmt.Println("Successfully attached eBPF uprobe to USDT tracepoint!")
	fmt.Println()

	// Get the perf event map
	eventsMap := coll.Maps["events"]
	if eventsMap == nil {
		log.Fatal("Events map not found in eBPF program")
	}

	// Create perf event reader
	reader, err := perf.NewReader(eventsMap, os.Getpagesize())
	if err != nil {
		log.Fatalf("Failed to create perf reader: %v", err)
	}
	defer reader.Close()

	fmt.Println("TIME                 KERNEL_ID   DURATION_MS")
	fmt.Println("==================== =========== ===========")

	// Set up signal handling
	ctx, cancel := context.WithCancel(context.Background())
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	go util.ReadTracePipe(ctx)

	go func() {
		<-sig
		fmt.Println("\nShutting down eBPF tracer...")
		cancel()
	}()

	// Monitor for events
	go func() {
		for {
			select {
			case <-ctx.Done():
				log.Printf("Context cancelled, stopping event reader")
				return
			default:
				log.Printf("Waiting for perf events...")
				record, err := reader.Read()
				if err != nil {
					if err == perf.ErrClosed {
						return
					}
					log.Printf("Error reading perf event: %v", err)
					continue
				}

				if len(record.RawSample) < 8 {
					log.Printf("Invalid event size: %d bytes", len(record.RawSample))
					continue
				}

				var timing KernelTiming
				timing.KernelID = binary.LittleEndian.Uint32(record.RawSample[0:4])
				timing.DurationBits = binary.LittleEndian.Uint32(record.RawSample[4:8])

				// Convert raw bits back to float32
				durationMS := math.Float32frombits(timing.DurationBits)

				timestamp := time.Now().Format("2006-01-02 15:04:05")
				fmt.Printf("%s %08x    %8.3f ms\n",
					timestamp, timing.KernelID, durationMS)
			}
		}
	}()

	// Check if process exists and simulate monitoring
	if err := checkProcessExists(pid); err != nil {
		log.Printf("Warning: %v", err)
	}

	fmt.Printf("Monitoring PID %d... (Press Ctrl+C to stop)\n", pid)
	fmt.Println("Ready to receive perf events from eBPF program")

	// Keep the program running
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := checkProcessExists(pid); err != nil {
				fmt.Printf("Target process %d no longer exists\n", pid)
				return
			}
		}
	}
}

// parseUSDTProbes reads USDT probe information from ELF .note.stapsdt section
func parseUSDTProbes(path string) ([]USDTProbe, error) {
	file, err := elf.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var probes []USDTProbe

	// Find .note.stapsdt section
	for _, section := range file.Sections {
		if section.Name == ".note.stapsdt" {
			data, err := section.Data()
			if err != nil {
				return nil, err
			}

			// Parse note entries
			offset := 0
			for offset < len(data) {
				if offset+12 > len(data) {
					break
				}

				// Note header: namesz(4) + descsz(4) + type(4)
				namesz := binary.LittleEndian.Uint32(data[offset : offset+4])
				descsz := binary.LittleEndian.Uint32(data[offset+4 : offset+8])
				noteType := binary.LittleEndian.Uint32(data[offset+8 : offset+12])
				offset += 12

				if noteType != 3 { // NT_STAPSDT
					// Skip this note
					nameEnd := offset + int((namesz+3)&^3) // align to 4 bytes
					descEnd := nameEnd + int((descsz+3)&^3)
					offset = descEnd
					continue
				}

				// Skip owner name (should be "stapsdt")
				nameEnd := offset + int((namesz+3)&^3)

				if nameEnd+int(descsz) > len(data) {
					break
				}

				// Parse descriptor
				desc := data[nameEnd : nameEnd+int(descsz)]
				if len(desc) < 24 { // 3 uint64 values
					offset = nameEnd + int((descsz+3)&^3)
					continue
				}

				location := binary.LittleEndian.Uint64(desc[0:8])
				base := binary.LittleEndian.Uint64(desc[8:16])
				semaphore := binary.LittleEndian.Uint64(desc[16:24])

				// Parse strings: provider\0probe\0arguments\0
				stringData := desc[24:]
				strings := strings.Split(string(stringData), "\x00")
				if len(strings) >= 3 {
					probe := USDTProbe{
						Provider:  strings[0],
						Name:      strings[1],
						Location:  location,
						Base:      base,
						Semaphore: semaphore,
						Arguments: strings[2],
					}
					probes = append(probes, probe)
				}

				offset = nameEnd + int((descsz+3)&^3)
			}
			break
		}
	}

	return probes, nil
}

// findLibraryBaseAddress finds where a shared library is loaded in process memory
func findLibraryBaseAddress(pid int, libPath string) (uint64, error) {
	mapsPath := fmt.Sprintf("/proc/%d/maps", pid)
	mapsData, err := os.ReadFile(mapsPath)
	if err != nil {
		return 0, err
	}

	mapsContent := string(mapsData)
	for _, line := range strings.Split(mapsContent, "\n") {
		if strings.Contains(line, libPath) || strings.Contains(line, "libparcagpu.so") {
			// Parse the memory range: "7f1234567000-7f1234568000 r-xp ..."
			fields := strings.Fields(line)
			if len(fields) < 1 {
				continue
			}

			// Check if this is an executable segment (r-xp)
			if len(fields) >= 2 && strings.Contains(fields[1], "x") {
				addrRange := strings.Split(fields[0], "-")
				if len(addrRange) != 2 {
					continue
				}

				// Parse the start address
				baseAddr, err := strconv.ParseUint(addrRange[0], 16, 64)
				if err != nil {
					continue
				}

				return baseAddr, nil
			}
		}
	}

	return 0, fmt.Errorf("library %s not found in process memory maps", libPath)
}

func checkProcessExists(pid int) error {
	if _, err := os.Stat(fmt.Sprintf("/proc/%d", pid)); os.IsNotExist(err) {
		return fmt.Errorf("process %d does not exist", pid)
	}
	return nil
}
