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
	Reserved uint32
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
	t   *testing.T
	reg *procctl.Registry
	c   *procctl.Client
	cmd *exec.Cmd
}

// start runs test_cupti_prof with args and env and waits for its region.
func start(t *testing.T, env []string, args ...string) *proc {
	t.Helper()
	bin := filepath.Join(root(), "build-local/bin/test_cupti_prof")
	lib := filepath.Join(root(), "build-local/lib/libparcagpucupti.so")
	cmd := exec.Command(bin, append([]string{lib, "--duration=60"}, args...)...)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+filepath.Join(root(), "build-local/lib"))
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

	p := &proc{t: t, reg: reg, cmd: cmd}
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
	if s.PCCapable != 0 {
		t.Errorf("pc_capable %d without PC sampling", s.PCCapable)
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
