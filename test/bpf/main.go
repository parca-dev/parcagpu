// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// Test program that loads the activity_parser BPF program, attaches it
// to parcagpu USDT probes in the target shared library, and logs kernel
// activities received through the ring buffer. Also captures cubin modules
// and resolves PC addresses to source lines using llvm-dwarfdump.
//
// Usage:
//
//	go generate ./...
//	go build -o activity_parser .
//	sudo ./activity_parser -pid <PID> -lib <path/to/libparcagpucupti.so>
package main

import (
	"bufio"
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	ebpf2 "github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/parca-dev/usdt"
	"golang.org/x/sys/unix"

	sasstable "github.com/gnurizen/sass-table"
)

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target $GOARCH -cflags "-I../../ebpf -I$USDT_HEADERS" activityParser activity_parser.bpf.c

// Event type tags — must match BPF #defines.
const (
	eventTypeKernel        = 1
	eventTypeCubinLoaded   = 2
	eventTypeCubinUnloaded = 3
	eventTypePCSample      = 4
	eventTypeError         = 5
	eventTypeNvtx          = 6
	eventTypeNvtxName      = 7
)

// KernelEvent matches struct kernel_event in the BPF program.
type KernelEvent struct {
	EventType     uint32
	_             uint32
	Start         uint64
	End           uint64
	CorrelationID uint32
	DeviceID      uint32
	StreamID      uint32
	GraphID       uint32
	GraphNodeID   uint64
	Name          [128]byte
}

// CubinEvent matches struct cubin_event in the BPF program.
type CubinEvent struct {
	EventType uint32
	_         uint32
	CubinCRC  uint64
	CubinPtr  uint64
	CubinSize uint64
}

// StallReason matches struct cupti_stall_reason in the BPF program.
type StallReason struct {
	Index   uint32
	Samples uint32
}

// ErrorEvent matches struct error_event in the BPF program.
type ErrorEvent struct {
	EventType uint32
	ErrorCode int32
	Message   [256]byte
	Component [64]byte
}

// PCSampleEvent matches struct pc_sample_event in the BPF program.
type PCSampleEvent struct {
	EventType        uint32
	StallReasonCount uint32
	CubinCRC         uint64
	PCOffset         uint64
	FunctionIndex    uint32
	CorrelationID    uint32
	FunctionName     [128]byte
	StallReasons     [64]StallReason
}

// NvtxEvent matches the fixed part of struct nvtx_event in the BPF program
// (the collector's struct nvtx_event); the NUL-terminated message follows it
// and the record ends with the message.
type NvtxEvent struct {
	EventType     uint32
	_             uint32
	TimestampNs   uint64
	Tid           uint32
	CorrelationID uint32
	DomainID      uint32
	Kind          uint16
	Flags         uint16
	Color         uint32
	DeviceID      uint32
	Payload       uint64
	RangeID       uint64
	MessagePtr    uint64
}

// NvtxNameEvent matches struct nvtx_name_event in the BPF program.
type NvtxNameEvent struct {
	EventType uint32
	Kind      uint32
	ID        uint64
	Name      [128]byte
}

const (
	statBatches       = 0
	statActivities    = 1
	statKernels       = 2
	statDrops         = 3
	statNvtxBatches   = 4
	statNvtxEvents    = 5
	statNvtxTruncated = 6
)

var nvtxKindNames = map[uint16]string{1: "push", 2: "pop", 3: "start", 4: "end", 5: "mark"}

// nvtxSummary aggregates NVTX events for the end-of-run report that
// test-nvtx-real.sh checks.
type nvtxSummary struct {
	events     uint64
	byKind     map[uint16]uint64
	byDomain   map[uint32]uint64
	byMessage  map[string]uint64
	tids       map[uint32]bool
	colored    uint64
	payloadSum uint64
	openRanges map[uint64]bool // RANGE_START ids without a matching END
	badEnds    uint64          // RANGE_END ids never started
	names      []NvtxNameEvent
}

func newNvtxSummary() *nvtxSummary {
	return &nvtxSummary{
		byKind:     map[uint16]uint64{},
		byDomain:   map[uint32]uint64{},
		byMessage:  map[string]uint64{},
		tids:       map[uint32]bool{},
		openRanges: map[uint64]bool{},
	}
}

func (n *nvtxSummary) add(ev *NvtxEvent, msg string) {
	n.events++
	n.byKind[ev.Kind]++
	n.byDomain[ev.DomainID]++
	if msg != "" {
		n.byMessage[msg]++
	}
	n.tids[ev.Tid] = true
	if ev.Color != 0 {
		n.colored++
	}
	n.payloadSum += ev.Payload
	switch ev.Kind {
	case 3:
		n.openRanges[ev.RangeID] = true
	case 4:
		if n.openRanges[ev.RangeID] {
			delete(n.openRanges, ev.RangeID)
		} else {
			n.badEnds++
		}
	}
}

func (n *nvtxSummary) print(objs *activityParserObjects) {
	var batches, recs, trunc, drops uint64
	for key, dst := range map[uint32]*uint64{
		statNvtxBatches: &batches, statNvtxEvents: &recs,
		statNvtxTruncated: &trunc, statDrops: &drops,
	} {
		objs.Stats.Lookup(&key, dst)
	}
	fmt.Printf("nvtx: events=%d bpf_records=%d batches=%d truncated=%d drops=%d\n",
		n.events, recs, batches, trunc, drops)
	for k := uint16(1); k <= 5; k++ {
		fmt.Printf("nvtx kind %s=%d\n", nvtxKindNames[k], n.byKind[k])
	}
	domainNames := map[uint64]string{}
	for _, nm := range n.names {
		if nm.Kind == 5 {
			domainNames[nm.ID] = cString(nm.Name[:])
		}
	}
	domains := make([]uint32, 0, len(n.byDomain))
	for d := range n.byDomain {
		domains = append(domains, d)
	}
	sort.Slice(domains, func(i, j int) bool { return domains[i] < domains[j] })
	for _, d := range domains {
		fmt.Printf("nvtx domain id=%d name=%q events=%d\n", d, domainNames[uint64(d)], n.byDomain[d])
	}
	msgs := make([]string, 0, len(n.byMessage))
	for m := range n.byMessage {
		msgs = append(msgs, m)
	}
	sort.Strings(msgs)
	for _, m := range msgs {
		fmt.Printf("nvtx message %q count=%d\n", m, n.byMessage[m])
	}
	fmt.Printf("nvtx threads=%d colored=%d payload_sum=%d open_ranges=%d bad_ends=%d\n",
		len(n.tids), n.colored, n.payloadSum, len(n.openRanges), n.badEnds)
	for _, nm := range n.names {
		fmt.Printf("nvtx name kind=%d id=0x%x name=%q\n", nm.Kind, nm.ID, cString(nm.Name[:]))
	}
}

// lineEntry is a single address→source mapping from the DWARF line table.
type lineEntry struct {
	addr uint64
	file string
	line int
}

// cubinStore holds loaded cubins and their parsed line tables.
type cubinStore struct {
	mu     sync.RWMutex
	cubins map[uint64]*cubinInfo // keyed by CRC
	pid    int
}

// textSection holds one .text._Zfuncname section from the cubin ELF.
type textSection struct {
	name string
	data []byte
}

type cubinInfo struct {
	crc     uint64
	size    uint64
	lines   []lineEntry   // sorted by addr
	files   []string      // file table from line table header
	archSM  int           // SM version from ELF e_flags (e.g. 121)
	texts   []textSection // .text sections for instruction decoding
	tmpPath string        // temp file for llvm-dwarfdump
}

func newCubinStore(pid int) *cubinStore {
	return &cubinStore{
		cubins: make(map[uint64]*cubinInfo),
		pid:    pid,
	}
}

func (cs *cubinStore) load(crc, ptr, size uint64) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	if _, ok := cs.cubins[crc]; ok {
		return // already loaded
	}

	data, err := readProcessMemory(cs.pid, ptr, size)
	if err != nil {
		log.Printf("  [CUBIN] failed to read cubin 0x%x (%d bytes) from pid %d: %v",
			crc, size, cs.pid, err)
		return
	}

	info := &cubinInfo{crc: crc, size: size}

	// Write to temp file for llvm-dwarfdump.
	tmp, err := os.CreateTemp("", fmt.Sprintf("cubin_%x_*.elf", crc))
	if err != nil {
		log.Printf("  [CUBIN] 0x%x loaded (%d bytes), no temp file: %v", crc, size, err)
		cs.cubins[crc] = info
		return
	}
	tmp.Write(data)
	tmp.Close()
	info.tmpPath = tmp.Name()

	// Parse line table with llvm-dwarfdump.
	lines, files, err := parseLinesWithDwarfdump(info.tmpPath)
	if err != nil {
		log.Printf("  [CUBIN] 0x%x loaded (%d bytes), no line info: %v", crc, size, err)
	} else {
		info.lines = lines
		info.files = files
		log.Printf("  [CUBIN] 0x%x loaded (%d bytes), %d line entries, %d files",
			crc, size, len(lines), len(files))
	}

	// Parse ELF to extract SM version and .text sections for SASS decoding.
	archSM, texts := parseCubinELF(data)
	info.archSM = archSM
	info.texts = texts
	if archSM > 0 {
		log.Printf("  [CUBIN] 0x%x SM%d, %d text sections", crc, archSM, len(texts))
	}

	cs.cubins[crc] = info
}

func (cs *cubinStore) unload(crc uint64) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	if info, ok := cs.cubins[crc]; ok {
		if info.tmpPath != "" {
			os.Remove(info.tmpPath)
		}
		delete(cs.cubins, crc)
	}
	log.Printf("  [CUBIN] 0x%x unloaded", crc)
}

func (cs *cubinStore) cleanup() {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	for _, info := range cs.cubins {
		if info.tmpPath != "" {
			os.Remove(info.tmpPath)
		}
	}
}

// resolveInstruction looks up the SASS mnemonic at a PC offset.
// Uses the sass-table opcode decoder first, falls back to nvdisasm cache.
func (cs *cubinStore) resolveInstruction(cubinCRC uint64, pcOffset uint64) string {
	cs.mu.RLock()
	info, ok := cs.cubins[cubinCRC]
	cs.mu.RUnlock()
	if !ok {
		return ""
	}

	if info.archSM == 0 || len(info.texts) == 0 {
		return ""
	}

	// pcOffset is function-relative. Try all text sections — the offset
	// should only produce a valid decode in the correct one.
	for _, ts := range info.texts {
		if int(pcOffset)+16 <= len(ts.data) {
			m := sasstable.DecodeMnemonicFromSlice(info.archSM, ts.data[pcOffset:])
			if m != "" {
				return m
			}
		}
	}
	return ""
}

// parseCubinELF extracts the SM version and .text section data from a cubin ELF.
func parseCubinELF(data []byte) (archSM int, texts []textSection) {
	f, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return 0, nil
	}
	defer f.Close()

	// Go's debug/elf doesn't expose e_flags. Read it directly from the ELF header.
	// For ELF64: e_flags is at offset 48, 4 bytes little-endian.
	if len(data) >= 52 {
		flags := binary.LittleEndian.Uint32(data[48:52])
		archSM = int((flags >> 8) & 0xFF)
	}

	for _, s := range f.Sections {
		if s.Type == elf.SHT_PROGBITS && s.Flags&elf.SHF_EXECINSTR != 0 &&
			strings.HasPrefix(s.Name, ".text") {
			d, err := s.Data()
			if err != nil {
				continue
			}
			texts = append(texts, textSection{name: s.Name, data: d})
		}
	}
	return archSM, texts
}

// resolvePC looks up a PC offset in a cubin's line table.
func (cs *cubinStore) resolvePC(cubinCRC uint64, pcOffset uint64) (file string, line int) {
	cs.mu.RLock()
	info, ok := cs.cubins[cubinCRC]
	cs.mu.RUnlock()
	if !ok || len(info.lines) == 0 {
		return
	}

	// Binary search for the largest address <= pcOffset.
	i := sort.Search(len(info.lines), func(i int) bool {
		return info.lines[i].addr > pcOffset
	})
	if i == 0 {
		return
	}
	e := info.lines[i-1]
	return e.file, e.line
}

// parseLinesWithDwarfdump runs llvm-dwarfdump --debug-line on the cubin ELF
// and parses the output into a sorted line table.
func parseLinesWithDwarfdump(path string) ([]lineEntry, []string, error) {
	cmd := exec.Command("llvm-dwarfdump", "--debug-line", path)
	out, err := cmd.Output()
	if err != nil {
		return nil, nil, fmt.Errorf("llvm-dwarfdump: %w", err)
	}

	var entries []lineEntry
	files := map[string]bool{}
	fileList := []string{}

	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := scanner.Text()

		// Parse file table entries: file_names[  N]:
		//            name: "foo.cu"
		if strings.HasPrefix(line, "           name: ") {
			name := strings.Trim(strings.TrimPrefix(line, "           name: "), "\"")
			if !files[name] {
				files[name] = true
				fileList = append(fileList, name)
			}
			continue
		}

		// Parse line table rows: 0xADDR  LINE  COL  FILE ...
		if !strings.HasPrefix(line, "0x") {
			continue
		}

		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}

		addr, err := strconv.ParseUint(strings.TrimPrefix(fields[0], "0x"), 16, 64)
		if err != nil {
			continue
		}

		lineNum, err := strconv.Atoi(fields[1])
		if err != nil {
			continue
		}

		// fields[3] is the file index (1-based)
		fileIdx, err := strconv.Atoi(fields[3])
		if err != nil || fileIdx < 1 || fileIdx > len(fileList) {
			continue
		}

		entries = append(entries, lineEntry{
			addr: addr,
			file: fileList[fileIdx-1],
			line: lineNum,
		})
	}

	if len(entries) == 0 {
		return nil, nil, fmt.Errorf("no line entries found")
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].addr < entries[j].addr
	})

	return entries, fileList, nil
}

func readProcessMemory(pid int, addr, size uint64) ([]byte, error) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/mem", pid))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	data := make([]byte, size)
	_, err = f.ReadAt(data, int64(addr))
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	return data, nil
}

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
		var ve *ebpf2.VerifierError
		if errors.As(err, &ve) {
			log.Fatalf("Verifier error loading BPF objects:\n%+v", ve)
		}
		log.Fatalf("Loading BPF objects: %v", err)
	}
	defer objs.Close()

	// Parse USDT probes from the shared library's .note.stapsdt section.
	probes, err := usdt.ParseProbesFromFile(realLib)
	if err != nil {
		log.Fatalf("Parsing USDT probes from %s: %v", realLib, err)
	}

	// Find USDT probes and attach uprobes at each site.
	ex, err := link.OpenExecutable(realLib)
	if err != nil {
		log.Fatalf("Opening executable %s: %v", realLib, err)
	}

	type probeTarget struct {
		name    string
		handler *ebpf2.Program
	}
	targets := []probeTarget{
		{"activity_batch", objs.HandleActivityBatch},
		{"stall_reason_map", objs.HandleStallReasonMap},
		{"cubin_loaded", objs.HandleCubinLoaded},
		{"cubin_unloaded", objs.HandleCubinUnloaded},
		{"pc_sample_batch", objs.HandlePcSampleBatch},
		{"error", objs.HandleError},
		{"nvtx_event_batch", objs.HandleNvtxEventBatch},
		{"nvtx_resource_name", objs.HandleNvtxResourceName},
	}

	var links []link.Link
	var specID uint32
	for _, t := range targets {
		for _, probe := range probes {
			if probe.Provider != "parcagpu" || probe.Name != t.name {
				continue
			}

			spec, err := usdt.ParseUSDTArguments(probe.Arguments)
			if err != nil {
				log.Fatalf("Parsing USDT args %q: %v", probe.Arguments, err)
			}

			specBytes := usdt.SpecToBytes(spec)
			if err := objs.BpfUsdtSpecs.Put(specID, specBytes); err != nil {
				log.Fatalf("Populating USDT spec map: %v", err)
			}

			cookie := uint64(specID) << 32

			log.Printf("USDT probe parcagpu:%s at offset 0x%x, args=%q, spec_id=%d",
				t.name, probe.Location, probe.Arguments, specID)

			up, err := ex.Uprobe(t.name, t.handler, &link.UprobeOptions{
				Address:      probe.Location,
				PID:          *pid,
				Cookie:       cookie,
				RefCtrOffset: probe.SemaphoreOffset,
			})
			if err != nil {
				log.Fatalf("Attaching uprobe for %s at offset 0x%x: %v", t.name, probe.Location, err)
			}
			links = append(links, up)
			specID++
		}
	}

	if len(links) == 0 {
		log.Fatalf("No parcagpu USDT probes found in %s", realLib)
	}
	defer func() {
		for _, l := range links {
			l.Close()
		}
	}()

	cubins := newCubinStore(*pid)
	defer cubins.cleanup()

	// Stall reason index → name cache, populated lazily from BPF map.
	stallReasonNames := map[uint32]string{}

	// Open ring buffer reader.
	rd, err := ringbuf.NewReader(objs.Events)
	if err != nil {
		log.Fatalf("Opening ring buffer: %v", err)
	}
	defer rd.Close()

	// Handle signals.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

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
			// Batches fired at exit (atexit / TLS destructors) may still be
			// in the ring buffer; let the reader drain them.
			time.Sleep(time.Second)
		}
		rd.Close()
	}()

	log.Printf("Attached %d USDT probe(s) in %s (PID %d)", len(links), realLib, *pid)
	log.Printf("Waiting for events...")

	var eventCount uint64
	var pcSampleCount uint64
	nvtx := newNvtxSummary()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	go func() {
		for range ticker.C {
			printStats(&objs, eventCount)
		}
	}()

	for {
		record, err := rd.Read()
		if err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				break
			}
			log.Printf("Reading from ring buffer: %v", err)
			continue
		}

		raw := record.RawSample
		if len(raw) < 4 {
			continue
		}

		eventType := binary.LittleEndian.Uint32(raw[:4])

		switch eventType {
		case eventTypeKernel:
			var event KernelEvent
			if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &event); err != nil {
				log.Printf("Parsing kernel event: %v", err)
				continue
			}
			eventCount++
			if *verbose {
				name := cString(event.Name[:])
				duration := event.End - event.Start
				fmt.Printf("kernel: name=%-40s corr=%-6d dev=%d stream=%d graph=%-3d duration=%dns\n",
					name, event.CorrelationID, event.DeviceID, event.StreamID, event.GraphID, duration)
			}

		case eventTypeCubinLoaded:
			var event CubinEvent
			if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &event); err != nil {
				log.Printf("Parsing cubin event: %v", err)
				continue
			}
			cubins.load(event.CubinCRC, event.CubinPtr, event.CubinSize)

		case eventTypeCubinUnloaded:
			var event CubinEvent
			if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &event); err != nil {
				log.Printf("Parsing cubin event: %v", err)
				continue
			}
			cubins.unload(event.CubinCRC)

		case eventTypeError:
			var event ErrorEvent
			if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &event); err != nil {
				log.Printf("Parsing error event: %v", err)
				continue
			}
			msg := cString(event.Message[:])
			comp := cString(event.Component[:])
			log.Printf("ERROR [%s] code=%d: %s", comp, event.ErrorCode, msg)

		case eventTypeNvtx:
			var event NvtxEvent
			if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &event); err != nil {
				log.Printf("Parsing nvtx event: %v", err)
				continue
			}
			msg := cString(raw[binary.Size(event):])
			nvtx.add(&event, msg)
			if *verbose {
				fmt.Printf("nvtx: kind=%s domain=%d tid=%d range=%d color=0x%x payload=%d msg=%q\n",
					nvtxKindNames[event.Kind], event.DomainID, event.Tid, event.RangeID,
					event.Color, event.Payload, msg)
			}

		case eventTypeNvtxName:
			var event NvtxNameEvent
			if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &event); err != nil {
				log.Printf("Parsing nvtx name event: %v", err)
				continue
			}
			nvtx.names = append(nvtx.names, event)

		case eventTypePCSample:
			var event PCSampleEvent
			if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &event); err != nil {
				log.Printf("Parsing pc sample event: %v", err)
				continue
			}
			pcSampleCount++

			// Lazily populate stall reason name cache.
			if len(stallReasonNames) == 0 {
				for i := uint32(0); i < 64; i++ {
					var name [64]byte
					if err := objs.StallReasons.Lookup(&i, &name); err != nil {
						continue
					}
					s := cString(name[:])
					if s != "" {
						stallReasonNames[i] = s
					}
				}
			}

			name := cString(event.FunctionName[:])
			file, line := cubins.resolvePC(event.CubinCRC, event.PCOffset)
			insn := cubins.resolveInstruction(event.CubinCRC, event.PCOffset)

			src := ""
			if file != "" {
				src = fmt.Sprintf("  %s:%d", file, line)
			}
			insnStr := ""
			if insn != "" {
				insnStr = fmt.Sprintf("  [%s]", insn)
			}
			corrStr := ""
			if event.CorrelationID != 0 {
				corrStr = fmt.Sprintf("  corr=%d", event.CorrelationID)
			}
			fmt.Printf("pc_sample: %s  pc=0x%04x%s%s%s\n", name, event.PCOffset, src, insnStr, corrStr)
			for i := uint32(0); i < event.StallReasonCount; i++ {
				sr := event.StallReasons[i]
				if sr.Samples == 0 {
					continue
				}
				srName := stallReasonNames[sr.Index]
				if srName == "" {
					srName = fmt.Sprintf("reason[%d]", sr.Index)
				}
				fmt.Printf("    %s = %d\n", srName, sr.Samples)
			}
		}
	}

	fmt.Println()
	log.Printf("Final stats:")
	printStats(&objs, eventCount)
	log.Printf("  pc_samples=%d", pcSampleCount)
	printStallReasonMap(&objs)
	printCubins(cubins)
	nvtx.print(&objs)
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

func printStallReasonMap(objs *activityParserObjects) {
	var loaded uint32
	loadedKey := uint32(0)
	if err := objs.StallMapLoaded.Lookup(&loadedKey, &loaded); err != nil || loaded == 0 {
		log.Printf("  stall reason map: not received")
		return
	}

	log.Printf("  stall reason map:")
	for i := uint32(0); i < 64; i++ {
		var name [64]byte
		if err := objs.StallReasons.Lookup(&i, &name); err != nil {
			continue
		}
		s := cString(name[:])
		if s == "" {
			continue
		}
		log.Printf("    [%2d] %s", i, s)
	}
}

func printCubins(cs *cubinStore) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	if len(cs.cubins) == 0 {
		log.Printf("  cubins: none loaded")
		return
	}

	log.Printf("  cubins loaded: %d", len(cs.cubins))
	for crc, info := range cs.cubins {
		log.Printf("    crc=0x%x size=%d lines=%d files=%v",
			crc, info.size, len(info.lines), info.files)

		// Print first 10 line entries as demo.
		for i, e := range info.lines {
			if i >= 10 {
				log.Printf("      ... and %d more entries", len(info.lines)-10)
				break
			}
			log.Printf("      0x%04x -> %s:%d", e.addr, e.file, e.line)
		}
	}
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

// Ensure struct sizes match BPF.
var _ = [1]struct{}{}[unsafe.Sizeof(KernelEvent{})-176]
var _ = [1]struct{}{}[unsafe.Sizeof(CubinEvent{})-32]
var _ = [1]struct{}{}[unsafe.Sizeof(PCSampleEvent{})-672]
