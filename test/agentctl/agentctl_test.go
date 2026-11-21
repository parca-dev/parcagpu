// Tests parcagpu's agent control (src/agent_ctl.cpp) from the agent side: a
// procctl registry, as parca-agent runs it, against test_cupti_prof with mock
// CUPTI. No GPU or root needed.
//
//	make test-agent-ctl
package agentctl

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/parca-dev/procctl"
)

// Mirrors src/parcagpu_ctl.h.
type config struct {
	Devices  uint32
	Factor   uint32
	WindowMs uint32
	GapMs    uint32
	// Launch probe rate limits.
	RateLimit           uint32
	GraphRateLimit      uint32
	Features            uint32
	HWBufferMB          uint32
	MetadataRefreshMs   uint32
	BlockingThresholdUs uint32
	Reserved            [22]uint32 // knobs to come
}

// PARCAGPU_FEATURE_*
const (
	featureKernelStats   = 1 << 0
	featureBlockingCalls = 1 << 1
)

type device struct {
	UUID           [16]byte
	SamplingFactor uint32
	ClockKHz       uint32
	SMCount        uint32
	PCState        uint32
}

type state struct {
	PCCapable      uint32
	RateLimit      uint32
	GraphRateLimit uint32
	NumDevices     uint32
	Devices        [16]device
	Features       uint32
	HWBufferMB     uint32
}

const (
	pcOff = iota
	pcOn
	pcUnavailable
)

func (c config) bytes() []byte {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, c)
	return b.Bytes()
}

func root() string {
	r, _ := filepath.Abs("../..")
	return r
}

// proc is a test_cupti_prof process under a procctl registry.
type proc struct {
	t         *testing.T
	reg       *procctl.Registry
	c         *procctl.Client
	cmd       *exec.Cmd
	stopAgent context.CancelFunc // stops heartbeats, as if the agent died
	mockState string             // MOCK_PC_STATE_FILE
	out       string             // test_cupti_prof's output
}

// start runs test_cupti_prof with args and env and waits for its region.
func start(t *testing.T, env []string, args ...string) *proc {
	t.Helper()
	bin := filepath.Join(root(), "build-local/bin/test_cupti_prof")
	lib := filepath.Join(root(), "build-local/lib/libparcagpucupti.so")
	mockState := filepath.Join(t.TempDir(), "mock_pc_state")
	cmd := exec.Command(bin, append([]string{lib, "--duration=60"}, args...)...)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+filepath.Join(root(), "build-local/lib"),
		"MOCK_PC_STATE_FILE="+mockState)
	cmd.Env = append(cmd.Env, env...)
	out, _ := os.Create(filepath.Join(t.TempDir(), "out.log"))
	cmd.Stdout, cmd.Stderr = out, out
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		if t.Failed() {
			b, _ := os.ReadFile(out.Name())
			if len(b) > 4096 {
				b = b[len(b)-4096:]
			}
			t.Logf("test_cupti_prof output (tail):\n%s", b)
		}
	})

	reg, err := procctl.NewRegistry(procctl.RegistryOptions{
		Consumer: "parcagpu", RescanInterval: 100 * time.Millisecond, Lease: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = reg.Run(ctx) }()
	t.Cleanup(func() { cancel(); _ = reg.Close() })

	p := &proc{t: t, reg: reg, cmd: cmd, stopAgent: cancel, mockState: mockState, out: out.Name()}
	p.eventually("region attached", func() error {
		for _, c := range reg.Clients() {
			if c.Info().HostPID == cmd.Process.Pid {
				p.c = c
				return nil
			}
		}
		return fmt.Errorf("pid %d not attached", cmd.Process.Pid)
	})
	return p
}

func (p *proc) eventually(what string, f func() error) {
	p.t.Helper()
	var err error
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		if err = f(); err == nil {
			return
		}
	}
	p.t.Fatalf("%s: %v", what, err)
}

func (p *proc) state() (state, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	st, err := p.c.ReadState(ctx)
	if err != nil {
		return state{}, err
	}
	var s state
	if err := binary.Read(bytes.NewReader(st.Consumer), binary.LittleEndian, &s); err != nil {
		return state{}, fmt.Errorf("consumer state (%d bytes): %w", len(st.Consumer), err)
	}
	return s, nil
}

// write sends cfg (nil: an empty payload) and reports whether parcagpu acked it.
func (p *proc) write(cfg *config, wait time.Duration) bool {
	p.t.Helper()
	var payload []byte
	if cfg != nil {
		payload = cfg.bytes()
	}
	gen, err := p.c.Write(payload)
	if err != nil {
		p.t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), wait)
	defer cancel()
	return p.c.WaitAck(ctx, gen) == nil
}

// sampling reports, per device slot, whether the mock samples its context now
// and how many times it was started (mock contexts are 1 + slot).
func (p *proc) sampling(slot int) (now bool, starts uint32) {
	b, err := os.ReadFile(p.mockState)
	i := 2 * (1 + slot) * 4
	if err != nil || len(b) < i+8 {
		p.t.Fatalf("mock state: %v (%d bytes)", err, len(b))
	}
	return binary.LittleEndian.Uint32(b[i:]) != 0, binary.LittleEndian.Uint32(b[i+4:])
}

func (p *proc) devices(n int) state {
	var s state
	p.eventually(fmt.Sprintf("%d devices", n), func() (err error) {
		if s, err = p.state(); err == nil && s.NumDevices != uint32(n) {
			err = fmt.Errorf("num_devices %d", s.NumDevices)
		}
		return err
	})
	return s
}

// grant writes cfg and checks it is acked, and that by then nothing is
// sampling outside it: parcagpu stops before it acks.
func (p *proc) grant(cfg config, slots int) {
	p.t.Helper()
	if !p.write(&cfg, 5*time.Second) {
		p.t.Fatalf("%+v not acked", cfg)
	}
	for i := 0; i < slots; i++ {
		if now, _ := p.sampling(i); now && cfg.Devices&(1<<i) == 0 {
			p.t.Errorf("%+v acked while slot %d still samples", cfg, i)
		}
		if p.enabled(i) && cfg.Devices&(1<<i) == 0 {
			p.t.Errorf("%+v acked while slot %d still has PC sampling enabled", cfg, i)
		}
	}
	s, err := p.state()
	if err != nil {
		p.t.Fatal(err)
	}
	for i := 0; i < slots; i++ {
		want := uint32(pcOff)
		if cfg.Devices&(1<<i) != 0 {
			want = pcOn
		}
		if got := s.Devices[i].PCState; got != want && got != pcUnavailable {
			p.t.Errorf("%+v: slot %d pc_state %d, want %d", cfg, i, got, want)
		}
	}
}

// bad reports the mock's count of starts of a started context and stops of
// a stopped one.
func (p *proc) bad() (starts, stops uint32) {
	b, err := os.ReadFile(p.mockState)
	i := 2 * 65 * 4
	if err != nil || len(b) < i+8 {
		p.t.Fatalf("mock state: %v (%d bytes)", err, len(b))
	}
	return binary.LittleEndian.Uint32(b[i:]), binary.LittleEndian.Uint32(b[i+4:])
}

// hwBuffer reports the hardware buffer size set on slot's context, MB (0: not
// set since PC sampling was enabled on it).
func (p *proc) hwBuffer(slot int) uint32 {
	b, err := os.ReadFile(p.mockState)
	i := (2*65 + 3 + 65 + 1 + slot) * 4
	if err != nil || len(b) < i+4 {
		p.t.Fatalf("mock state: %v (%d bytes)", err, len(b))
	}
	return binary.LittleEndian.Uint32(b[i:])
}

// enabled reports whether PC sampling is enabled on slot's context.
func (p *proc) enabled(slot int) bool {
	b, err := os.ReadFile(p.mockState)
	i := (2*65 + 3 + 1 + slot) * 4
	if err != nil || len(b) < i+4 {
		p.t.Fatalf("mock state: %v (%d bytes)", err, len(b))
	}
	return binary.LittleEndian.Uint32(b[i:]) != 0
}

// getContextIDs reports how many times the process called cuptiGetContextId.
func (p *proc) getContextIDs() uint32 {
	b, err := os.ReadFile(p.mockState)
	i := (2*65 + 2) * 4
	if err != nil || len(b) < i+4 {
		p.t.Fatalf("mock state: %v (%d bytes)", err, len(b))
	}
	return binary.LittleEndian.Uint32(b[i:])
}

// starts reports, per slot, how many times sampling started in d.
func (p *proc) startsIn(d time.Duration, slots int) []uint32 {
	before := make([]uint32, slots)
	for i := range before {
		_, before[i] = p.sampling(i)
	}
	time.Sleep(d)
	for i := range before {
		_, n := p.sampling(i)
		before[i] = n - before[i]
	}
	return before
}

// sampleAll makes every window open: the dice-roll probability starts at 1
// for a high target rate. It also turns on PC sampling by environment.
var sampleAll = []string{"PARCAGPU_PC_SAMPLING_RATE=1000000"}

func TestDeviceTable(t *testing.T) {
	p := start(t, nil, "--num-gpus=3")
	var s state
	p.eventually("3 devices", func() (err error) {
		if s, err = p.state(); err == nil && s.NumDevices != 3 {
			err = fmt.Errorf("num_devices %d", s.NumDevices)
		}
		return err
	})
	for i, d := range s.Devices[:3] {
		want := [16]byte{'M', 'O', 'C', 'K', 'G', 'P', 'U', '-', 15: byte(i)}
		if d.UUID != want || d.ClockKHz != 1830000 || d.SMCount != 132 || d.PCState != pcOff {
			t.Errorf("device %d: %+v", i, d)
		}
	}
	if s.PCCapable != 1 {
		t.Errorf("pc_capable %d", s.PCCapable)
	}
	// Not set up for PC sampling until the agent grants a device.
	for i, d := range s.Devices[:3] {
		if d.SamplingFactor != 0 || p.enabled(i) {
			t.Errorf("device %d: sampling_factor %d, enabled %v", i, d.SamplingFactor, p.enabled(i))
		}
	}
}

func TestConfigAck(t *testing.T) {
	p := start(t, nil, "--num-gpus=2")
	p.eventually("2 devices", func() (err error) {
		s, err := p.state()
		if err == nil && s.NumDevices != 2 {
			err = fmt.Errorf("num_devices %d", s.NumDevices)
		}
		return err
	})
	for _, tc := range []struct {
		name string
		cfg  *config
		ack  bool
	}{
		{"valid", &config{Devices: 3, Factor: 10, WindowMs: 50}, true},
		{"defaults", &config{}, true},
		{"empty payload", nil, true},
		{"factor too small", &config{Factor: 4}, false},
		{"factor too large", &config{Factor: 32}, false},
		{"window too long", &config{WindowMs: 10001}, false},
		{"unknown device", &config{Devices: 1 << 2}, false},
		{"rate limit too high", &config{RateLimit: 1000001}, false},
		{"graph rate limit too high", &config{GraphRateLimit: 1000001}, false},
		{"hw buffer too large", &config{HWBufferMB: 4097}, false},
		{"largest hw buffer", &config{HWBufferMB: 4096}, true},
		{"any metadata refresh period", &config{MetadataRefreshMs: 0xffffffff}, true},
		{"blocking threshold too high", &config{BlockingThresholdUs: 1000001}, false},
		{"highest blocking threshold", &config{BlockingThresholdUs: 1000000}, true},
		{"knobs it doesn't know", &config{Devices: 1, Reserved: [22]uint32{0: 1, 21: 7}}, true},
		{"valid again", &config{Devices: 1, Factor: 31, WindowMs: 10000}, true},
	} {
		wait := 5 * time.Second
		if !tc.ack {
			wait = 500 * time.Millisecond
		}
		if got := p.write(tc.cfg, wait); got != tc.ack {
			t.Errorf("%s: acked %v, want %v", tc.name, got, tc.ack)
		}
	}

	// A config of another size isn't one this schema describes.
	for _, n := range []int{24, 132} {
		gen, err := p.c.Write(make([]byte, n))
		if err != nil {
			continue // procctl refuses more than the payload capacity
		}
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		if p.c.WaitAck(ctx, gen) == nil {
			t.Errorf("%d-byte config acked", n)
		}
		cancel()
	}
}

// The agent's mask decides which devices sample; a revoked device has stopped
// by the time parcagpu acks.
func TestGrantRevoke(t *testing.T) {
	p := start(t, sampleAll, "--num-gpus=2", "--threads=2", "--launch-rate=2000")
	p.devices(2)
	for _, mask := range []uint32{1, 2, 3, 0, 1} {
		p.grant(config{Devices: mask, WindowMs: 50}, 2)
		got := p.startsIn(time.Second, 2)
		for i, n := range got {
			if granted := mask&(1<<i) != 0; granted != (n > 0) {
				t.Errorf("mask %b: slot %d started %d times in 1s", mask, i, n)
			}
		}
	}
}

// Revokes race with windows opening and closing on many threads: every
// start and stop must still pair up.
func TestGrantRevokeStress(t *testing.T) {
	p := start(t, sampleAll, "--num-gpus=4", "--threads=8", "--launch-rate=5000")
	p.devices(4)
	for i := 0; i < 50; i++ {
		p.grant(config{Devices: uint32(i*7) % 16, WindowMs: uint32(1 + i%5)}, 4)
		time.Sleep(20 * time.Millisecond)
	}
	if s, e := p.bad(); s != 0 || e != 0 {
		t.Errorf("bad starts %d, bad stops %d", s, e)
	}
}

// A granted device is enabled for PC sampling on its next launch; a revoked
// one is disabled before the ack (grant checks), so another process can
// take the GPU's sampler.
func TestEnableOnGrant(t *testing.T) {
	p := start(t, nil, "--num-gpus=2", "--launch-rate=2000")
	p.devices(2)
	for _, mask := range []uint32{1, 2, 0, 3, 0} {
		p.grant(config{Devices: mask, WindowMs: 5}, 2)
		p.eventually(fmt.Sprintf("enabled per %b", mask), func() error {
			for i := 0; i < 2; i++ {
				if p.enabled(i) != (mask&(1<<i) != 0) {
					return fmt.Errorf("slot %d enabled %v", i, p.enabled(i))
				}
			}
			return nil
		})
	}
}

// Under the agent the window and the gap set the duty cycle, not the dice.
func TestDutyCycle(t *testing.T) {
	// No env sampling: the dice would open ~nothing.
	p := start(t, nil, "--num-gpus=1", "--threads=4", "--launch-rate=4000")
	p.devices(1)
	p.grant(config{Devices: 1, WindowMs: 10, GapMs: 90}, 1)
	time.Sleep(500 * time.Millisecond) // set up
	if n := p.startsIn(2*time.Second, 1)[0]; n < 10 || n > 22 {
		t.Errorf("window 10ms + gap 90ms: %d windows in 2s, want ~20", n)
	}
	p.grant(config{Devices: 1, WindowMs: 10, GapMs: 490}, 1)
	if n := p.startsIn(2*time.Second, 1)[0]; n < 2 || n > 6 {
		t.Errorf("window 10ms + gap 490ms: %d windows in 2s, want ~4", n)
	}
}

// A process that may not sample makes no CUPTI calls on its launches:
// cuptiGetContextId there corrupts the heap on a GB10 when another process
// has PC sampling enabled on the GPU (see PCSampling::contextIds).
func TestNoCUPTICallsWhenRevoked(t *testing.T) {
	p := start(t, sampleAll, "--num-gpus=2", "--threads=2", "--launch-rate=2000")
	p.devices(2)
	p.grant(config{Devices: 0}, 2)
	time.Sleep(200 * time.Millisecond)
	before := p.getContextIDs()
	time.Sleep(time.Second)
	if n := p.getContextIDs() - before; n != 0 {
		t.Errorf("%d cuptiGetContextId calls in 1s with no device granted", n)
	}
}

// A process that makes no CUDA calls still gives a device up promptly: the
// release thread does it.
func TestReleaseThread(t *testing.T) {
	// CUDA calls 500 ms apart (bursts of 5 launches), and windows that stay
	// open across them: a revoke within 300 ms is the release thread's.
	p := start(t, sampleAll, "--num-gpus=1", "--launch-rate=2")
	p.devices(1)
	for i := 0; i < 5; i++ {
		p.grant(config{Devices: 1, WindowMs: 10000}, 1)
		p.eventually("sampling", func() error {
			if now, _ := p.sampling(0); !now {
				return fmt.Errorf("not sampling")
			}
			return nil
		})
		start := time.Now()
		p.grant(config{Devices: 0}, 1)
		if d := time.Since(start); d > 300*time.Millisecond {
			t.Errorf("revoke took %v", d)
		}
	}
}

// A device another process samples reports UNAVAILABLE.
func TestBusyDevice(t *testing.T) {
	p := start(t, append(sampleAll, "MOCK_PC_UNKNOWN_CTX=1"), "--num-gpus=2", "--launch-rate=2000")
	p.devices(2)
	p.grant(config{Devices: 3, WindowMs: 50}, 2)
	p.eventually("slot 0 unavailable", func() error {
		s, err := p.state()
		if err == nil && (s.Devices[0].PCState != pcUnavailable || s.Devices[1].PCState != pcOn) {
			err = fmt.Errorf("pc_state %d %d", s.Devices[0].PCState, s.Devices[1].PCState)
		}
		return err
	})
}

// A factor change re-configures granted devices, live.
func TestFactor(t *testing.T) {
	p := start(t, sampleAll, "--num-gpus=2", "--launch-rate=2000")
	p.devices(2)
	factors := func(want0, want1 uint32) func() error {
		return func() error {
			s, err := p.state()
			if err == nil && (s.Devices[0].SamplingFactor != want0 || s.Devices[1].SamplingFactor != want1) {
				err = fmt.Errorf("factors %d %d", s.Devices[0].SamplingFactor, s.Devices[1].SamplingFactor)
			}
			return err
		}
	}
	p.grant(config{Devices: 1, Factor: 12, WindowMs: 50}, 2)
	// Slot 1 was set up at the default (20) by the environment before the
	// agent took over; disabled since, it reports what it last used.
	p.eventually("slot 0 at 12", factors(12, 20))
	p.grant(config{Devices: 3, Factor: 15, WindowMs: 50}, 2)
	p.eventually("both at 15", factors(15, 15))
	p.grant(config{Devices: 3, WindowMs: 50}, 2)
	p.eventually("both back to the default", factors(20, 20))
}

// A hardware buffer size change applies at a context's next setup: a granted
// device is set up again, live; one not granted keeps what it was set up with
// until granted. The state reports the size in effect.
func TestHWBuffer(t *testing.T) {
	p := start(t, append([]string{"PARCAGPU_PC_HW_BUFFER_MB=32"}, sampleAll...),
		"--num-gpus=2", "--launch-rate=2000")
	p.devices(2)
	sizes := func(wantState, want0, want1 uint32) func() error {
		return func() error {
			s, err := p.state()
			if err != nil {
				return err
			}
			if s.HWBufferMB != wantState || p.hwBuffer(0) != want0 || p.hwBuffer(1) != want1 {
				err = fmt.Errorf("state %d, set %d %d", s.HWBufferMB, p.hwBuffer(0), p.hwBuffer(1))
			}
			return err
		}
	}
	// Set up by the environment when created.
	p.eventually("the environment's", sizes(32, 32, 32))
	p.grant(config{Devices: 1, HWBufferMB: 64, WindowMs: 50}, 2)
	// Slot 1 was disabled when the agent took it away.
	p.eventually("slot 0 at 64", sizes(64, 64, 0))
	p.grant(config{Devices: 3, HWBufferMB: 64, WindowMs: 50}, 2)
	p.eventually("both at 64", sizes(64, 64, 64))
	p.grant(config{Devices: 3, HWBufferMB: 128, WindowMs: 50}, 2)
	p.eventually("both at 128", sizes(128, 128, 128))
	p.grant(config{Devices: 3, WindowMs: 50}, 2)
	p.eventually("both back to the environment's", sizes(32, 32, 32))
	p.stopAgent()
	p.eventually("the environment's after lease expiry", sizes(32, 32, 32))
}

// Without PARCAGPU_PC_HW_BUFFER_MB, 0 leaves CUPTI's default: the attribute
// isn't set at all.
func TestHWBufferDefault(t *testing.T) {
	p := start(t, sampleAll, "--num-gpus=1", "--launch-rate=2000")
	p.devices(1)
	size := func(wantState, want uint32) func() error {
		return func() error {
			s, err := p.state()
			if err == nil && (s.HWBufferMB != wantState || p.hwBuffer(0) != want) {
				err = fmt.Errorf("state %d, set %d", s.HWBufferMB, p.hwBuffer(0))
			}
			return err
		}
	}
	p.grant(config{Devices: 1, HWBufferMB: 16, WindowMs: 50}, 1)
	p.eventually("the agent's", size(16, 16))
	p.grant(config{Devices: 1, WindowMs: 50}, 1)
	p.eventually("CUPTI's default", size(0, 0))
}

// refreshesIn reports how many metadata refreshes parcagpu logged in d
// (PARCAGPU_DEBUG=1).
func (p *proc) refreshesIn(d time.Duration) int {
	count := func() int {
		b, _ := os.ReadFile(p.out)
		return bytes.Count(b, []byte("metadata refresh (period"))
	}
	before := count()
	time.Sleep(d)
	return count() - before
}

// The agent's metadata refresh period replaces the environment's while it
// sets one, from the next launch: lowering it doesn't wait out the old period.
func TestMetadataRefresh(t *testing.T) {
	p := start(t, []string{"PARCAGPU_DEBUG=1", "PARCAGPU_METADATA_REFRESH_MS=60000"},
		"--num-gpus=1", "--launch-rate=2000")
	p.devices(1)
	slow := func(what string) {
		p.eventually(what, func() error {
			if n := p.refreshesIn(time.Second); n > 1 {
				return fmt.Errorf("%d refreshes in 1 s", n)
			}
			return nil
		})
	}
	slow("the environment's 60 s")
	if !p.write(&config{MetadataRefreshMs: 100}, 5*time.Second) {
		t.Fatal("not acked")
	}
	// At once: within a second of the ack, not after the rest of 60 s.
	if n := p.refreshesIn(time.Second); n < 5 {
		t.Errorf("agent's 100 ms: %d refreshes in 1 s", n)
	}
	if !p.write(&config{}, 5*time.Second) {
		t.Fatal("not acked")
	}
	slow("the environment's again")
	if !p.write(&config{MetadataRefreshMs: 100}, 5*time.Second) {
		t.Fatal("not acked")
	}
	p.stopAgent()
	slow("the environment's after lease expiry")
}

// blockingIn reports the weights, ns, of the cuda_blocking_call events
// parcagpu logged in d (PARCAGPU_DEBUG=1), and how many were their call's
// duration.
func (p *proc) blockingIn(d time.Duration) (weights []uint64, exact int) {
	read := func() []byte { b, _ := os.ReadFile(p.out); return b }
	before := len(read())
	time.Sleep(d)
	for _, m := range blockingRe.FindAllSubmatch(read()[before:], -1) {
		dur, _ := strconv.ParseUint(string(m[1]), 10, 64)
		w, _ := strconv.ParseUint(string(m[2]), 10, 64)
		weights = append(weights, w)
		if w == dur {
			exact++
		}
	}
	return weights, exact
}

var blockingRe = regexp.MustCompile(`cuda_blocking_call: cbid=-?\d+ dev=\d+ duration=(\d+)ns weight=(\d+)ns`)

// The agent's cuda_blocking_call threshold replaces the environment's while
// it sets one, from the next call; it is kept while blocking calls are off.
func TestBlockingThreshold(t *testing.T) {
	// 50 us syncs; the environment's threshold is 1 ms.
	p := start(t, []string{"PARCAGPU_DEBUG=1", "PARCAGPU_BLOCKING_THRESHOLD_US=1000"},
		"--num-gpus=1", "--launch-rate=200", "--sync-us=50")
	p.devices(1)
	// Every weight is the threshold (ns): calls are shorter than it. Some
	// are kept unless they rarely are (minEvents 0).
	atThreshold := func(what string, ns uint64, minEvents int) {
		p.t.Helper()
		w, _ := p.blockingIn(time.Second)
		if len(w) < minEvents {
			t.Errorf("%s: %d events", what, len(w))
		}
		for _, x := range w {
			if x != ns {
				t.Errorf("%s: weight %d, want %d", what, x, ns)
				return
			}
		}
	}
	// Every call kept, at its duration: calls are longer than the threshold.
	// Checked right after the ack: it applies from the next call.
	allKept := func(what string) {
		p.t.Helper()
		if w, exact := p.blockingIn(time.Second); len(w) < 20 || exact != len(w) {
			t.Errorf("%s: %d events, %d at their duration", what, len(w), exact)
		}
	}
	// Set while blocking calls are off: kept, without effect.
	if !p.write(&config{BlockingThresholdUs: 10}, 5*time.Second) {
		t.Fatal("not acked")
	}
	if w, _ := p.blockingIn(time.Second); len(w) != 0 {
		t.Errorf("blocking calls off: %d events", len(w))
	}
	if !p.write(&config{Features: featureBlockingCalls, BlockingThresholdUs: 10}, 5*time.Second) {
		t.Fatal("not acked")
	}
	allKept("the agent's 10 us")
	if !p.write(&config{Features: featureBlockingCalls}, 5*time.Second) {
		t.Fatal("not acked")
	}
	atThreshold("the environment's 1 ms", 1_000_000, 1)
	if !p.write(&config{Features: featureBlockingCalls, BlockingThresholdUs: 1}, 5*time.Second) {
		t.Fatal("not acked")
	}
	allKept("the agent's 1 us")
	if !p.write(&config{Features: featureBlockingCalls, BlockingThresholdUs: 1000000}, 5*time.Second) {
		t.Fatal("not acked")
	}
	atThreshold("the agent's 1 s", 1_000_000_000, 0)
}

// When the agent's lease expires, parcagpu goes back to its environment:
// PC sampling everywhere if PARCAGPU_PC_SAMPLING_RATE is set, else nowhere.
func TestLeaseExpiry(t *testing.T) {
	for _, tc := range []struct {
		name string
		env  []string
	}{{"env sampling", sampleAll}, {"no env sampling", nil}} {
		t.Run(tc.name, func(t *testing.T) {
			p := start(t, tc.env, "--num-gpus=2", "--launch-rate=2000")
			p.devices(2)
			p.grant(config{Devices: 1, WindowMs: 1}, 2)
			p.eventually("slot 0 sampling", func() error {
				if got := p.startsIn(500*time.Millisecond, 2); got[0] == 0 || got[1] != 0 {
					return fmt.Errorf("starts %v", got)
				}
				return nil
			})
			p.stopAgent()
			p.eventually("environment behaviour", func() error {
				got := p.startsIn(500*time.Millisecond, 2)
				if (tc.env != nil) != (got[0] > 0 && got[1] > 0) || (tc.env == nil) != (got[0] == 0 && got[1] == 0) {
					return fmt.Errorf("starts %v", got)
				}
				return nil
			})
		})
	}
}

// The agent's launch probe rate limits replace the environment's while it
// sets them; parcagpu reports those in effect.
func TestRateLimits(t *testing.T) {
	p := start(t, []string{"PARCAGPU_RATE_LIMIT=50"}, "--num-gpus=1")
	p.devices(1)
	rates := func(want, wantGraph uint32) func() error {
		return func() error {
			s, err := p.state()
			if err == nil && (s.RateLimit != want || s.GraphRateLimit != wantGraph) {
				err = fmt.Errorf("rate limits %d %d", s.RateLimit, s.GraphRateLimit)
			}
			return err
		}
	}
	p.eventually("the environment's", rates(50, 20))
	if !p.write(&config{RateLimit: 1000, GraphRateLimit: 5}, 5*time.Second) {
		t.Fatal("not acked")
	}
	p.eventually("the agent's", rates(1000, 5))
	if !p.write(&config{GraphRateLimit: 7}, 5*time.Second) {
		t.Fatal("not acked")
	}
	p.eventually("environment's kernel launch rate", rates(50, 7))
	p.stopAgent()
	p.eventually("the environment's after lease expiry", rates(50, 20))
}

// Kernel stats follow the agent's feature bit while it controls the process,
// and are on (for any consumer attached to the probe) without it.
func TestKernelStatsFeature(t *testing.T) {
	p := start(t, nil, "--num-gpus=1")
	p.devices(1)
	features := func(want uint32) func() error {
		return func() error {
			s, err := p.state()
			if err == nil && s.Features != want {
				err = fmt.Errorf("features %#x", s.Features)
			}
			return err
		}
	}
	p.eventually("on without an agent", features(featureKernelStats))
	if !p.write(&config{}, 5*time.Second) {
		t.Fatal("not acked")
	}
	p.eventually("off by the agent's default", features(0))
	if !p.write(&config{Features: featureKernelStats | 1<<7}, 5*time.Second) {
		t.Fatal("not acked")
	}
	p.eventually("on by the agent; unknown bits ignored", features(featureKernelStats))
	p.stopAgent()
	p.eventually("on after lease expiry", features(featureKernelStats))
}

// cuda_blocking_call follows PARCAGPU_BLOCKING_CALLS without an agent, and
// the agent's bit under it.
func TestBlockingCallsFeature(t *testing.T) {
	features := func(p *proc, want uint32) func() error {
		return func() error {
			s, err := p.state()
			if err == nil && s.Features != want {
				err = fmt.Errorf("features %#x", s.Features)
			}
			return err
		}
	}
	t.Run("environment off", func(t *testing.T) {
		p := start(t, nil, "--num-gpus=1")
		p.devices(1)
		p.eventually("off without the environment", features(p, featureKernelStats))
		if !p.write(&config{Features: featureBlockingCalls}, 5*time.Second) {
			t.Fatal("not acked")
		}
		p.eventually("on by the agent", features(p, featureBlockingCalls))
	})
	t.Run("environment on", func(t *testing.T) {
		p := start(t, []string{"PARCAGPU_BLOCKING_CALLS=1"}, "--num-gpus=1")
		p.devices(1)
		p.eventually("on by the environment", features(p, featureKernelStats|featureBlockingCalls))
		if !p.write(&config{}, 5*time.Second) {
			t.Fatal("not acked")
		}
		p.eventually("off by the agent's default", features(p, 0))
		p.stopAgent()
		p.eventually("the environment's after lease expiry", features(p, featureKernelStats|featureBlockingCalls))
	})
}
