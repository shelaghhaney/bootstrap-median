// Command bootstrap estimates the standard error of the median via bootstrap
// resampling for three distributional shapes (symmetric, positively skewed,
// negatively skewed). It can either generate its own synthetic data or read
// the CSV files produced by r_code/run-bootstrap-median.R, enabling a true
// apples-to-apples performance comparison on identical observations.
//
// Usage:
//
//	bootstrap [-n <obs>] [-b <resamples>] [-seed <uint32>] [-fromdata <dir>] [-profile] [-log <file>]
//
// Flags:
//
//	-n          number of observations per dataset when generating data (default 500)
//	-b          number of bootstrap resamples (default 1000)
//	-seed       MT19937 seed (default 42)
//	-fromdata   directory containing symmetric.csv, positively_skewed.csv,
//	            and negatively_skewed.csv written by the R script; when set,
//	            -n and -seed are ignored for data generation
//	-profile    write CPU and memory profiles to results/
//	-log        path for log file (default results/bootstrap.log)
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/shelaghhaney/bootstrap-median/internal/bootstrap"
	"github.com/shelaghhaney/bootstrap-median/internal/logging"
	"github.com/shelaghhaney/bootstrap-median/internal/profiling"
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

// dataSource describes one dataset: a human-readable label, the distribution
// shape (used when generating data), and the CSV filename (used when loading
// data written by R).
type dataSource struct {
	label   string
	shape   bootstrap.DistributionShape
	csvFile string // filename inside the -fromdata directory
}

// allSources defines the three distributions analysed in both R and Go.
var allSources = []dataSource{
	{
		label:   "Symmetric (Normal)",
		shape:   bootstrap.ShapeSymmetric,
		csvFile: "symmetric.csv",
	},
	{
		label:   "Positively Skewed (Log-Normal)",
		shape:   bootstrap.ShapePositivelySkewed,
		csvFile: "positively_skewed.csv",
	},
	{
		label:   "Negatively Skewed (Reflected Log-Normal)",
		shape:   bootstrap.ShapeNegativelySkewed,
		csvFile: "negatively_skewed.csv",
	},
}

func main() {
	// ── Flags ──────────────────────────────────────────────────────────────
	nObs      := flag.Int("n", 500, "observations per dataset (ignored when -fromdata is set)")
	nResample := flag.Int("b", 1000, "number of bootstrap resamples")
	seed      := flag.Uint("seed", 42, "MT19937 seed for data generation and resampling")
	fromData  := flag.String("fromdata", "", "directory of CSV files written by the R script")
	doProfile := flag.Bool("profile", false, "write CPU/memory profiles to results/")
	logPath   := flag.String("log", filepath.Join("results", "bootstrap.log"), "path for log file")
	flag.Parse()

	if *nResample < 1 {
		fmt.Fprintln(os.Stderr, "error: -b must be >= 1")
		os.Exit(1)
	}
	if *fromData == "" && *nObs < 2 {
		fmt.Fprintln(os.Stderr, "error: -n must be >= 2")
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

	dataMode := "generated"
	if *fromData != "" {
		dataMode = fmt.Sprintf("loaded from %s", *fromData)
	}
	logger.Info("Bootstrap Median SE Estimator")
	logger.Info("Parameters: B=%d  seed=%d  data=%s", *nResample, *seed, dataMode)

	// ── Optional CPU profiling ──────────────────────────────────────────────
	if *doProfile {
		stop, profErr := profiling.StartCPU(filepath.Join("results", "cpu.prof"))
		if profErr != nil {
			logger.Warn("CPU profile unavailable: %v", profErr)
		} else {
			defer stop()
			logger.Info("CPU profiling active → results/cpu.prof")
		}
	}

	// ── Bootstrap config ────────────────────────────────────────────────────
	cfg := bootstrap.Config{
		NumResamples: *nResample,
		Seed:         uint32(*seed),
	}

	// MT19937 used only when generating synthetic data (not needed for -fromdata)
	rng := bootstrap.NewMT19937(uint32(*seed))

	// ── Header ──────────────────────────────────────────────────────────────
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Bootstrap Standard Error of the Median – Go (MT19937)")
	if *fromData != "" {
		fmt.Printf("  Data: loaded from %s\n", *fromData)
	} else {
		fmt.Printf("  Data: generated  n=%d  seed=%d\n", *nObs, *seed)
	}
	fmt.Printf("  B=%d resamples\n", *nResample)
	fmt.Println("══════════════════════════════════════════════════════════════════════")

	// ── Main loop ───────────────────────────────────────────────────────────
	var records []runRecord

	for _, src := range allSources {
		// Obtain data either from CSV or by generation
		var data []float64
		if *fromData != "" {
			csvPath := filepath.Join(*fromData, src.csvFile)
			data, err = loadCSV(csvPath)
			if err != nil {
				logger.Error("Cannot load %s: %v", csvPath, err)
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
				os.Exit(1)
			}
			logger.Info("Loaded %d observations from %s", len(data), csvPath)
		} else {
			data = bootstrap.GenerateDataset(rng, src.shape, *nObs)
			logger.Info("Generated %d %s observations", *nObs, src.label)
		}

		memBefore := profiling.TakeMemSnapshot()
		start := time.Now()

		result, runErr := bootstrap.Run(data, bootstrap.Median, cfg)
		elapsed := time.Since(start)

		if runErr != nil {
			logger.Error("Bootstrap failed for %s: %v", src.label, runErr)
			continue
		}

		memAfter := profiling.TakeMemSnapshot()
		allocKB := float64(memAfter.TotalAllocBytes-memBefore.TotalAllocBytes) / 1024.0

		logger.Info("%s | n=%d  median=%.4f  SE=%.4f  bias=%.4f  95%%CI=[%.4f,%.4f]  elapsed=%s  alloc=%.1fKB",
			src.label, len(data),
			result.OriginalEstimate, result.BootstrapSE, result.Bias,
			result.CI95Lower, result.CI95Upper,
			elapsed, allocKB)

		fmt.Printf("\n  Distribution: %s\n", src.label)
		fmt.Printf("  ─────────────────────────────────────────────────\n")
		fmt.Printf("  Observations (n):          %8d\n", len(data))
		fmt.Printf("  Sample Median (original):  %8.4f\n", result.OriginalEstimate)
		fmt.Printf("  Bootstrap SE of Median:    %8.4f\n", result.BootstrapSE)
		fmt.Printf("  Bootstrap Bias:            %8.4f\n", result.Bias)
		fmt.Printf("  95%% Percentile CI:         [%7.4f, %7.4f]\n", result.CI95Lower, result.CI95Upper)
		fmt.Printf("  Bootstrap resamples (B):   %8d\n", result.NumResamples)
		fmt.Printf("  Elapsed time:              %v\n", elapsed)
		fmt.Printf("  Memory allocated (delta):  %.1f KB\n", allocKB)

		records = append(records, runRecord{
			Shape:     src.label,
			N:         len(data),
			B:         *nResample,
			Estimate:  result.OriginalEstimate,
			SE:        result.BootstrapSE,
			Bias:      result.Bias,
			CI95Lo:    result.CI95Lower,
			CI95Hi:    result.CI95Upper,
			ElapsedMs: float64(elapsed.Microseconds()) / 1000.0,
			AllocKB:   allocKB,
		})

		// Save raw bootstrap distribution for each shape
		distPath := filepath.Join("results", fmt.Sprintf("boot_dist_%s.csv", src.shape))
		if saveErr := saveFloat64CSV(distPath, result.Resamples); saveErr != nil {
			logger.Warn("Could not save bootstrap distribution: %v", saveErr)
		}
	}

	// ── Summary CSV ─────────────────────────────────────────────────────────
	summaryPath := filepath.Join("results", "go_results.csv")
	if saveErr := saveSummaryCSV(summaryPath, records); saveErr != nil {
		logger.Error("Could not write summary CSV: %v", saveErr)
	} else {
		logger.Info("Results saved to %s", summaryPath)
	}

	// ── Memory profile ──────────────────────────────────────────────────────
	if *doProfile {
		if profErr := profiling.WriteMemProfile(filepath.Join("results", "mem.prof")); profErr != nil {
			logger.Warn("Memory profile write failed: %v", profErr)
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

// loadCSV reads a single-column CSV file (with a header row named "value")
// as written by R's write.csv() call. It returns an error if the file is
// missing, unreadable, or contains fewer than 2 numeric rows.
func loadCSV(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("loadCSV: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)

	// Skip the header row written by R's write.csv()
	if _, err := r.Read(); err != nil {
		return nil, fmt.Errorf("loadCSV: cannot read header of %s: %w", path, err)
	}

	var values []float64
	lineNum := 1
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		lineNum++
		if err != nil {
			return nil, fmt.Errorf("loadCSV: parse error at line %d of %s: %w", lineNum, path, err)
		}
		if len(record) == 0 {
			continue
		}
		v, parseErr := strconv.ParseFloat(record[0], 64)
		if parseErr != nil {
			return nil, fmt.Errorf("loadCSV: non-numeric value %q at line %d of %s", record[0], lineNum, path)
		}
		values = append(values, v)
	}

	if len(values) < 2 {
		return nil, fmt.Errorf("loadCSV: %s has only %d observations; need at least 2", path, len(values))
	}
	return values, nil
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
	for _, rec := range records {
		row := []string{
			rec.Shape,
			strconv.Itoa(rec.N),
			strconv.Itoa(rec.B),
			strconv.FormatFloat(rec.Estimate, 'f', 6, 64),
			strconv.FormatFloat(rec.SE, 'f', 6, 64),
			strconv.FormatFloat(rec.Bias, 'f', 6, 64),
			strconv.FormatFloat(rec.CI95Lo, 'f', 6, 64),
			strconv.FormatFloat(rec.CI95Hi, 'f', 6, 64),
			strconv.FormatFloat(rec.ElapsedMs, 'f', 3, 64),
			strconv.FormatFloat(rec.AllocKB, 'f', 2, 64),
		}
		_ = w.Write(row)
	}
	w.Flush()
	return w.Error()
}
