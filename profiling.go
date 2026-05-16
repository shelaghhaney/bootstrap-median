// Package profiling wraps Go's runtime/pprof and runtime.MemStats to provide
// easy-to-use CPU and memory profiling for the bootstrap application.
package profiling

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

// CPUProfile starts CPU profiling to outPath and returns a stop function.
// Usage:
//
//	stop, err := profiling.StartCPU("cpu.prof")
//	defer stop()
func StartCPU(outPath string) (stop func(), err error) {
	f, err := os.Create(outPath)
	if err != nil {
		return func() {}, fmt.Errorf("profiling: cannot create CPU profile file: %w", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		return func() {}, fmt.Errorf("profiling: cannot start CPU profile: %w", err)
	}
	return func() {
		pprof.StopCPUProfile()
		f.Close()
	}, nil
}

// WriteMemProfile writes a heap/memory profile to outPath.
func WriteMemProfile(outPath string) error {
	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("profiling: cannot create memory profile file: %w", err)
	}
	defer f.Close()
	runtime.GC() // force a GC to get accurate statistics
	return pprof.WriteHeapProfile(f)
}

// MemSnapshot holds a snapshot of key runtime memory statistics.
type MemSnapshot struct {
	AllocBytes      uint64
	TotalAllocBytes uint64
	SysBytes        uint64
	NumGC           uint32
	Timestamp       time.Time
}

// TakeMemSnapshot captures current runtime memory stats.
func TakeMemSnapshot() MemSnapshot {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return MemSnapshot{
		AllocBytes:      ms.Alloc,
		TotalAllocBytes: ms.TotalAlloc,
		SysBytes:        ms.Sys,
		NumGC:           ms.NumGC,
		Timestamp:       time.Now(),
	}
}

// FormatMemSnapshot returns a human-readable summary of a MemSnapshot.
func FormatMemSnapshot(label string, s MemSnapshot) string {
	return fmt.Sprintf(
		"%s | Alloc=%.2f KB  TotalAlloc=%.2f KB  Sys=%.2f KB  NumGC=%d",
		label,
		float64(s.AllocBytes)/1024,
		float64(s.TotalAllocBytes)/1024,
		float64(s.SysBytes)/1024,
		s.NumGC,
	)
}
