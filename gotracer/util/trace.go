package util

import (
	"bufio"
	"context"
	"errors"
	"io"
	"log"
	"os"
	"strings"
)

func getTracePipe() (*os.File, error) {
	for _, mnt := range []string{
		"/sys/kernel/debug/tracing",
		"/sys/kernel/tracing",
		"/tracing",
		"/trace"} {
		t, err := os.Open(mnt + "/trace_pipe")
		if err == nil {
			return t, nil
		}
		log.Printf("Could not open trace_pipe at %s: %s", mnt, err)
	}
	return nil, os.ErrNotExist
}

func ReadTracePipe(ctx context.Context) {
	tp, err := getTracePipe()
	if err != nil {
		log.Printf("Could not open trace_pipe, check that debugfs is mounted")
		return
	}

	// When we're done kick ReadString out of blocked I/O.
	go func() {
		<-ctx.Done()
		tp.Close()
	}()

	r := bufio.NewReader(tp)

	log.Println("reading trace_pipe")
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				continue
			}
			log.Print(err)
			return
		}
		line = strings.TrimSpace(line)
		if len(line) > 0 {
			log.Printf("%s\n", line)
		}
	}
}
