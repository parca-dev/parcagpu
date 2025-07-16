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
	"reflect"
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

// Manually enable a perf event by extracting the FD from perfEventLink
// Note: BPF program attachment is already handled by Cilium eBPF library
func enablePerfEvent(l link.Link) error {
	// Use reflection to access the unexported fields
	linkValue := reflect.ValueOf(l)
	if linkValue.Kind() == reflect.Ptr {
		linkValue = linkValue.Elem()
	}

	// Check if this is a perfEventLink
	if linkValue.Type().Name() != "perfEventLink" {
		return fmt.Errorf("not a perfEventLink")
	}

	// Get the pe field (perfEvent)
	peField := linkValue.FieldByName("pe")
	if !peField.IsValid() {
		return fmt.Errorf("pe field not found")
	}

	// Access the fd field from perfEvent
	peValue := peField.Elem() // dereference the pointer
	fdField := peValue.FieldByName("fd")
	if !fdField.IsValid() {
		return fmt.Errorf("fd field not found")
	}

	// Extract the file descriptor from sys.FD struct
	// sys.FD has a raw int field
	fdValue := fdField.Elem() // dereference the *sys.FD
	rawField := fdValue.FieldByName("raw")
	if !rawField.IsValid() {
		return fmt.Errorf("raw field not found in sys.FD")
	}

	fd := int(rawField.Int())

	// Call PERF_EVENT_IOC_ENABLE (the missing piece!)
	const PERF_EVENT_IOC_ENABLE = 0x2400
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), PERF_EVENT_IOC_ENABLE, 0)
	if errno != 0 {
		return fmt.Errorf("ioctl PERF_EVENT_IOC_ENABLE failed: %v", errno)
	}

	return nil
}

// ProcessInfo contains information about a process with libparcagpu.so loaded
type ProcessInfo struct {
	PID      int
	LibPath  string
	BaseAddr uint64
}

// AttachedProcess represents a process we've attached to
type AttachedProcess struct {
	ProcessInfo
	Links []link.Link
}

// findProcessesWithLibrary scans all processes to find those with libparcagpu.so loaded
func findProcessesWithLibrary() ([]ProcessInfo, error) {
	var processes []ProcessInfo

	// Read /proc to get all PIDs
	procDir, err := os.Open("/proc")
	if err != nil {
		return nil, fmt.Errorf("failed to open /proc: %v", err)
	}
	defer procDir.Close()

	entries, err := procDir.Readdir(-1)
	if err != nil {
		return nil, fmt.Errorf("failed to read /proc: %v", err)
	}

	for _, entry := range entries {
		// Skip non-numeric directories (not PIDs)
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}

		// Read /proc/PID/maps to check for libparcagpu.so
		mapsPath := fmt.Sprintf("/proc/%d/maps", pid)
		mapsData, err := os.ReadFile(mapsPath)
		if err != nil {
			// Process might have terminated or we don't have permission
			continue
		}

		mapsContent := string(mapsData)
		for _, line := range strings.Split(mapsContent, "\n") {
			if strings.Contains(line, "libparcagpu.so") {
				fields := strings.Fields(line)
				if len(fields) >= 6 {
					libPath := fields[5]

					// Get base address for executable segment
					if strings.Contains(fields[1], "x") {
						addrRange := strings.Split(fields[0], "-")
						if len(addrRange) == 2 {
							baseAddr, err := strconv.ParseUint(addrRange[0], 16, 64)
							if err == nil {
								processes = append(processes, ProcessInfo{
									PID:      pid,
									LibPath:  libPath,
									BaseAddr: baseAddr,
								})
								break
							}
						}
					}
				}
			}
		}
	}

	return processes, nil
}

func main() {
	fmt.Println("eBPF USDT Tracer for parcagpu kernel_launch probe")
	fmt.Println("Scanning all processes for libparcagpu.so...")

	// Find all processes with libparcagpu.so
	processes, err := findProcessesWithLibrary()
	if err != nil {
		log.Fatalf("Failed to scan processes: %v", err)
	}

	if len(processes) == 0 {
		fmt.Println("No processes found with libparcagpu.so loaded")
		fmt.Println("Waiting for processes...")
	}

	fmt.Printf("Found %d process(es) with libparcagpu.so:\n", len(processes))
	for _, proc := range processes {
		fmt.Printf("  PID %d: %s (base: 0x%x)\n", proc.PID, proc.LibPath, proc.BaseAddr)
	}

	// Remove memory limit for eBPF
	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatalf("Failed to remove memlock limit: %v", err)
	}

	fmt.Printf("\nLoading eBPF program: %s\n", bpfObjectPath)

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

	// Debug: Print all available programs
	fmt.Println("Available programs in eBPF collection:")
	for name, _ := range coll.Programs {
		fmt.Printf("  - %s\n", name)
	}

	// Get the eBPF program for attaching
	prog := coll.Programs["usdt__trace_kernel_launch"]
	if prog == nil {
		log.Fatal("usdt__trace_kernel_launch program not found in eBPF collection")
	}

	// Attach to all processes
	var attachedProcesses []AttachedProcess

	for _, procInfo := range processes {
		fmt.Printf("\nAttaching to PID %d...\n", procInfo.PID)

		// For cross-container access, prepend /proc/<pid>/root to the path
		procLibPath := procInfo.LibPath
		if !strings.HasPrefix(procInfo.LibPath, "/proc/") {
			procLibPath = fmt.Sprintf("/proc/%d/root%s", procInfo.PID, procInfo.LibPath)
		}

		// Open the library executable
		ex, err := link.OpenExecutable(procLibPath)
		if err != nil {
			fmt.Printf("Failed to open library for PID %d: %v\n", procInfo.PID, err)
			continue
		}

		// Parse USDT probes from ELF file
		probes, err := parseUSDTProbes(procLibPath)
		if err != nil {
			fmt.Printf("Failed to parse USDT probes for PID %d: %v\n", procInfo.PID, err)
			continue
		}

		// Find all parcagpu:kernel_launch probes
		var targetProbes []USDTProbe
		for _, probe := range probes {
			if probe.Provider == "parcagpu" && probe.Name == "kernel_launch" {
				targetProbes = append(targetProbes, probe)
			}
		}

		if len(targetProbes) == 0 {
			fmt.Printf("No parcagpu:kernel_launch probes found for PID %d\n", procInfo.PID)
			continue
		}

		fmt.Printf("Found %d USDT probe locations for parcagpu:kernel_launch\n", len(targetProbes))

		// Attach uprobe to all USDT probe locations
		var links []link.Link
		for _, probe := range targetProbes {
			opts := &link.UprobeOptions{
				Address:      probe.Location,
				PID:          0xffffffff, // System-wide, like bpftrace
				RefCtrOffset: probe.Semaphore,
			}

			fmt.Printf("USDT probe details: location=0x%x, base=0x%x, semaphore=0x%x, process_base=0x%x\n",
				probe.Location, probe.Base, probe.Semaphore, procInfo.BaseAddr)

			l, err := ex.Uprobe("parcagpu:kernel_launch", prog, opts)
			if err != nil {
				fmt.Printf("Failed to attach to USDT probe at offset 0x%x: %v\n", probe.Location, err)
				continue
			}

			fmt.Printf("Uprobe attached successfully, type: %T\n", l)

			// Manually enable the perf event (BPF program already attached by Cilium)
			if err := enablePerfEvent(l); err != nil {
				fmt.Printf("Failed to enable perf event: %v\n", err)
				l.Close()
				continue
			}
			fmt.Printf("Perf event enabled successfully\n")

			links = append(links, l)
		}

		if len(links) > 0 {
			attachedProcesses = append(attachedProcesses, AttachedProcess{
				ProcessInfo: procInfo,
				Links:       links,
			})
			fmt.Printf("Successfully attached %d probe(s) to PID %d\n", len(links), procInfo.PID)
		}
	}

	// Defer closing all links
	defer func() {
		for _, ap := range attachedProcesses {
			for _, l := range ap.Links {
				l.Close()
			}
		}
	}()

	if len(attachedProcesses) > 0 {
		fmt.Printf("\nSuccessfully attached to %d process(es)!\n", len(attachedProcesses))
	} else {
		fmt.Println("\nNo processes attached. Will monitor for new processes...")
	}

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

	fmt.Println("\nTIME                 PID       KERNEL_ID   DURATION_MS")
	fmt.Println("==================== ========= =========== ===========")

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

				log.Printf("Received perf event! Size: %d bytes", len(record.RawSample))

				if len(record.RawSample) < 8 {
					log.Printf("Invalid event size: %d bytes", len(record.RawSample))
					continue
				}

				var timing KernelTiming
				timing.KernelID = binary.LittleEndian.Uint32(record.RawSample[0:4])
				timing.DurationBits = binary.LittleEndian.Uint32(record.RawSample[4:8])

				// Convert raw bits back to float32
				durationMS := math.Float32frombits(timing.DurationBits)

				// TODO: We could enhance this to show which PID generated the event
				// by including PID info in the eBPF program
				timestamp := time.Now().Format("2006-01-02 15:04:05")
				fmt.Printf("%s %-9s %08x    %8.3f ms\n",
					timestamp, "*", timing.KernelID, durationMS)
			}
		}
	}()

	fmt.Println("\nMonitoring for kernel launches... (Press Ctrl+C to stop)")
	fmt.Println("Ready to receive perf events from eBPF program")

	// Keep the program running and periodically check for new processes
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Check if attached processes still exist
			for i := len(attachedProcesses) - 1; i >= 0; i-- {
				ap := attachedProcesses[i]
				if err := checkProcessExists(ap.PID); err != nil {
					fmt.Printf("\nProcess %d no longer exists, detaching...\n", ap.PID)
					for _, l := range ap.Links {
						l.Close()
					}
					// Remove from slice
					attachedProcesses = append(attachedProcesses[:i], attachedProcesses[i+1:]...)
				}
			}

			// Look for new processes
			newProcesses, _ := findProcessesWithLibrary()
			for _, newProc := range newProcesses {
				// Check if we're already attached to this PID
				alreadyAttached := false
				for _, ap := range attachedProcesses {
					if ap.PID == newProc.PID {
						alreadyAttached = true
						break
					}
				}

				if !alreadyAttached {
					fmt.Printf("\nFound new process with libparcagpu.so: PID %d\n", newProc.PID)
					// Attach to new process
					procLibPath := newProc.LibPath
					if !strings.HasPrefix(newProc.LibPath, "/proc/") {
						procLibPath = fmt.Sprintf("/proc/%d/root%s", newProc.PID, newProc.LibPath)
					}

					ex, err := link.OpenExecutable(procLibPath)
					if err != nil {
						continue
					}

					probes, err := parseUSDTProbes(procLibPath)
					if err != nil {
						continue
					}

					var targetProbes []USDTProbe
					for _, probe := range probes {
						if probe.Provider == "parcagpu" && probe.Name == "kernel_launch" {
							targetProbes = append(targetProbes, probe)
						}
					}

					var links []link.Link
					for _, probe := range targetProbes {
						opts := &link.UprobeOptions{
							Address:      probe.Location,
							PID:          newProc.PID,
							RefCtrOffset: probe.Semaphore,
						}

						l, err := ex.Uprobe("parcagpu:kernel_launch", prog, opts)
						if err == nil {
							// Manually enable the perf event (BPF program already attached by Cilium)
							if err := enablePerfEvent(l); err != nil {
								fmt.Printf("Failed to enable perf event for new process: %v\n", err)
								l.Close()
								continue
							}
							links = append(links, l)
						}
					}

					if len(links) > 0 {
						attachedProcesses = append(attachedProcesses, AttachedProcess{
							ProcessInfo: newProc,
							Links:       links,
						})
						fmt.Printf("Successfully attached %d probe(s) to new PID %d\n", len(links), newProc.PID)
					}
				}
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
