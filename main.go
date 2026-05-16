// Command bootstrap estimates the standard error of the median via bootstrap
// resampling for three distributional shapes (symmetric, positively skewed,
// negatively skewed). It mirrors the R analysis in r_code/run-bootstrap-median.R
// and is intended for direct performance comparison.
//
// Usage:
//
//	bootstrap [-n <obs>] [-b <resamples>] [-seed <uint32>] [-profile] [-log <file>]
//
// Flags:
//
//	-n        number of observations per dataset (default 500)
//	-b        number of bootstrap resamples (default 1000)
//	-seed     MT19937 seed (default 42)
//	-profile  write CPU and memory profiles to results/
//	-log      path for log file (default results/bootstrap.log)
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/bootstrap-median/internal/bootstrap"
	"github.com/bootstrap-median/internal/logging"
	"github.com/bootstrap-median/internal/profiling"
)

// runRecord holds summary statistics for a single bootstrap run.
type runRecord struct {
	Shape     string
	N         int
	B         int
	Estimate  float64
	SE        float64
	Bias      float64
	CI95Lo    float64
	CI95Hi    float64
	ElapsedMs float64
	AllocKB   float64
}

func main() {
	// ── Flags ──────────────────────────────────────────────────────────────
	nObs      := flag.Int("n", 500, "number of observations per dataset")
	nResample := flag.Int("b", 1000, "number of bootstrap resamples")
	seed      := flag.Uint("seed", 42, "MT19937 random seed")
	doProfile := flag.Bool("profile", false, "write CPU/memory profiles to results/")
	logPath   := flag.String("log", filepath.Join("results", "bootstrap.log"), "path for log file")
	flag.Parse()

	if *nObs < 2 {
		fmt.Fprintln(os.Stderr, "error: -n must be >= 2")
		os.Exit(1)
	}
	if *nResample < 1 {
		fmt.Fprintln(os.Stderr, "error: -b must be >= 1")
		os.Exit(1)
	}

	// ── Setup logging ───────────────────────────────────────────────────────
	if err := os.MkdirAll("results", 0755); err != nil {
		fmt.Fprintln(os.Stderr, "cannot create results dir:", err)
		os.Exit(1)
	}
	logger, logFile, err := logging.NewFileLogger(*logPath, logging.INFO)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer logFile.Close()

	logger.Info("Bootstrap Median SE Estimator")
	logger.Info("Parameters: n=%d  B=%d  seed=%d", *nObs, *nResample, *seed)

	// ── Optional CPU profiling ──────────────────────────────────────────────
	if *doProfile {
		stop, err := profiling.StartCPU(filepath.Join("results", "cpu.prof"))
		if err != nil {
			logger.Warn("CPU profile unavailable: %v", err)
		} else {
			defer stop()
			logger.Info("CPU profiling active → results/cpu.prof")
		}
	}

	// ── Data generation ─────────────────────────────────────────────────────
	rng := bootstrap.NewMT19937(uint32(*seed))
	cfg := bootstrap.Config{
		NumResamples: *nResample,
		Seed:         uint32(*seed),
	}

	shapes := []struct {
		name  string
		shape bootstrap.DistributionShape
	}{
		{"Symmetric (Normal)", bootstrap.ShapeSymmetric},
		{"Positively Skewed (Log-Normal)", bootstrap.ShapePositivelySkewed},
		{"Negatively Skewed (Reflected Log-Normal)", bootstrap.ShapeNegativelySkewed},
	}

	var records []runRecord

	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Bootstrap Standard Error of the Median – Go (MT19937)")
	fmt.Printf("  n=%d observations  B=%d resamples  seed=%d\n", *nObs, *nResample, *seed)
	fmt.Println("══════════════════════════════════════════════════════════════════════")

	for _, s := range shapes {
		logger.Info("Generating %d %s observations", *nObs, s.name)
		memBefore := profiling.TakeMemSnapshot()

		data := bootstrap.GenerateDataset(rng, s.shape, *nObs)
		start := time.Now()
		result, err := bootstrap.Run(data, bootstrap.Median, cfg)
		elapsed := time.Since(start)

		if err != nil {
			logger.Error("Bootstrap failed for %s: %v", s.name, err)
			continue
		}

		memAfter := profiling.TakeMemSnapshot()
		allocDelta := float64(memAfter.TotalAllocBytes-memBefore.TotalAllocBytes) / 1024.0

		logger.Info("%s | median=%.4f  SE=%.4f  bias=%.4f  95%%CI=[%.4f, %.4f]  elapsed=%s  alloc=%.1fKB",
			s.name,
			result.OriginalEstimate, result.BootstrapSE, result.Bias,
			result.CI95Lower, result.CI95Upper,
			elapsed, allocDelta)

		fmt.Printf("\n  Distribution: %s\n", s.name)
		fmt.Printf("  ─────────────────────────────────────────────────\n")
		fmt.Printf("  Sample Median (original):  %8.4f\n", result.OriginalEstimate)
		fmt.Printf("  Bootstrap SE of Median:    %8.4f\n", result.BootstrapSE)
		fmt.Printf("  Bootstrap Bias:            %8.4f\n", result.Bias)
		fmt.Printf("  95%% Percentile CI:         [%7.4f, %7.4f]\n", result.CI95Lower, result.CI95Upper)
		fmt.Printf("  Bootstrap resamples:       %d\n", result.NumResamples)
		fmt.Printf("  Elapsed time:              %v\n", elapsed)
		fmt.Printf("  Memory allocated (delta):  %.1f KB\n", allocDelta)

		records = append(records, runRecord{
			Shape:     s.name,
			N:         *nObs,
			B:         *nResample,
			Estimate:  result.OriginalEstimate,
			SE:        result.BootstrapSE,
			Bias:      result.Bias,
			CI95Lo:    result.CI95Lower,
			CI95Hi:    result.CI95Upper,
			ElapsedMs: float64(elapsed.Microseconds()) / 1000.0,
			AllocKB:   allocDelta,
		})

		// Save raw bootstrap distribution for each shape
		distPath := filepath.Join("results", fmt.Sprintf("boot_dist_%s.csv", s.shape))
		if err := saveFloat64CSV(distPath, result.Resamples); err != nil {
			logger.Warn("Could not save bootstrap distribution: %v", err)
		}
	}

	// ── Save summary CSV ────────────────────────────────────────────────────
	summaryPath := filepath.Join("results", "go_results.csv")
	if err := saveSummaryCSV(summaryPath, records); err != nil {
		logger.Error("Could not write summary CSV: %v", err)
	} else {
		logger.Info("Results saved to %s", summaryPath)
	}

	// ── Memory profile ──────────────────────────────────────────────────────
	if *doProfile {
		if err := profiling.WriteMemProfile(filepath.Join("results", "mem.prof")); err != nil {
			logger.Warn("Memory profile write failed: %v", err)
		} else {
			logger.Info("Memory profile → results/mem.prof")
		}
	}

	snap := profiling.TakeMemSnapshot()
	logger.Info(profiling.FormatMemSnapshot("Final memory", snap))
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Log:     %s\n", *logPath)
	fmt.Printf("  Results: %s\n", summaryPath)
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println()
}

// saveFloat64CSV writes a slice of float64 as a single-column CSV.
func saveFloat64CSV(path string, values []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{"bootstrap_estimate"})
	for _, v := range values {
		_ = w.Write([]string{strconv.FormatFloat(v, 'f', 8, 64)})
	}
	w.Flush()
	return w.Error()
}

// saveSummaryCSV writes the summary records table to a CSV file.
func saveSummaryCSV(path string, records []runRecord) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	header := []string{"shape", "n", "B", "median_estimate", "bootstrap_se",
		"bias", "ci95_lower", "ci95_upper", "elapsed_ms", "alloc_kb"}
	_ = w.Write(header)
	for _, r := range records {
		row := []string{
			r.Shape,
			strconv.Itoa(r.N),
			strconv.Itoa(r.B),
			strconv.FormatFloat(r.Estimate, 'f', 6, 64),
			strconv.FormatFloat(r.SE, 'f', 6, 64),
			strconv.FormatFloat(r.Bias, 'f', 6, 64),
			strconv.FormatFloat(r.CI95Lo, 'f', 6, 64),
			strconv.FormatFloat(r.CI95Hi, 'f', 6, 64),
			strconv.FormatFloat(r.ElapsedMs, 'f', 3, 64),
			strconv.FormatFloat(r.AllocKB, 'f', 2, 64),
		}
		_ = w.Write(row)
	}
	w.Flush()
	return w.Error()
}
