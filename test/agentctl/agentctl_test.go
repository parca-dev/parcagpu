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
}

type device struct {
	UUID           [16]byte
	SamplingFactor uint32
	ClockKHz       uint32
	SMCount        uint32
	PCState        uint32
}

type state struct {
	PCCapable  uint32
	NumDevices uint32
	Devices    [16]device
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

	p := &proc{t: t, reg: reg, cmd: cmd, stopAgent: cancel, mockState: mockState}
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
	// Slot 1 was set up at the default (10) by the environment before the
	// agent took over; disabled since, it reports what it last used.
	p.eventually("slot 0 at 12", factors(12, 10))
	p.grant(config{Devices: 3, Factor: 15, WindowMs: 50}, 2)
	p.eventually("both at 15", factors(15, 15))
	p.grant(config{Devices: 3, WindowMs: 50}, 2)
	p.eventually("both back to the default", factors(10, 10))
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
