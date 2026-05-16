// Package bootstrap implements distribution-free bootstrap resampling for
// estimating the standard error of statistical estimators. The method follows
// Efron & Tibshirani (1993) and is the Go analogue of R's boot package.
//
// Selected method: Bootstrapping and simulation-based inference (Gelman & Vehtari 2021).
// R packages: boot (https://cran.r-project.org/package=boot),
//             bootstrap (https://cran.r-project.org/package=bootstrap).
package bootstrap

import (
	"fmt"
	"math"
	"sort"
)

// Config holds all parameters governing a bootstrap run.
type Config struct {
	NumResamples int    // number of bootstrap resamples (B)
	Seed         uint32 // MT19937 seed for reproducibility
}

// DefaultConfig returns a Config with reasonable defaults matching R's boot().
func DefaultConfig() Config {
	return Config{
		NumResamples: 1000,
		Seed:         42,
	}
}

// Result summarises a completed bootstrap analysis.
type Result struct {
	OriginalEstimate float64   // estimator applied to the full sample
	BootstrapSE      float64   // bootstrap standard error
	BootstrapMean    float64   // mean of the bootstrap distribution
	Bias             float64   // bootstrap bias = BootstrapMean - OriginalEstimate
	CI95Lower        float64   // 2.5th percentile (basic percentile CI)
	CI95Upper        float64   // 97.5th percentile
	Resamples        []float64 // all bootstrap estimates (for diagnostics)
	NumResamples     int
}

// EstimatorFunc is any function that maps a slice of float64 to a scalar.
// Examples: Median, Mean, Variance, trimmed mean, etc.
type EstimatorFunc func(data []float64) float64

// Run performs bootstrap resampling of estimatorFn on data using cfg.
// It is safe to call concurrently with different data slices.
func Run(data []float64, estimatorFn EstimatorFunc, cfg Config) (Result, error) {
	n := len(data)
	if n < 2 {
		return Result{}, fmt.Errorf("bootstrap: data must have at least 2 observations, got %d", n)
	}
	if cfg.NumResamples < 1 {
		return Result{}, fmt.Errorf("bootstrap: NumResamples must be >= 1, got %d", cfg.NumResamples)
	}

	rng := NewMT19937(cfg.Seed)
	original := estimatorFn(data)

	bootEstimates := make([]float64, cfg.NumResamples)
	resample := make([]float64, n) // reuse allocation across resamples

	for b := 0; b < cfg.NumResamples; b++ {
		// Draw n observations with replacement
		for i := 0; i < n; i++ {
			resample[i] = data[rng.Intn(n)]
		}
		bootEstimates[b] = estimatorFn(resample)
	}

	se, mean := stddevAndMean(bootEstimates)
	sort.Float64s(bootEstimates)

	lo := percentile(bootEstimates, 2.5)
	hi := percentile(bootEstimates, 97.5)

	return Result{
		OriginalEstimate: original,
		BootstrapSE:      se,
		BootstrapMean:    mean,
		Bias:             mean - original,
		CI95Lower:        lo,
		CI95Upper:        hi,
		Resamples:        bootEstimates,
		NumResamples:     cfg.NumResamples,
	}, nil
}

// Median returns the sample median of data (does not modify the slice).
func Median(data []float64) float64 {
	if len(data) == 0 {
		return math.NaN()
	}
	tmp := make([]float64, len(data))
	copy(tmp, data)
	sort.Float64s(tmp)
	n := len(tmp)
	if n%2 == 1 {
		return tmp[n/2]
	}
	return (tmp[n/2-1] + tmp[n/2]) / 2.0
}

// Mean returns the arithmetic mean of data.
func Mean(data []float64) float64 {
	if len(data) == 0 {
		return math.NaN()
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// stddevAndMean computes population standard deviation and mean in one pass
// using Welford's online algorithm (numerically stable).
func stddevAndMean(values []float64) (stddev, mean float64) {
	n := float64(len(values))
	if n == 0 {
		return math.NaN(), math.NaN()
	}
	var m, s float64
	for i, x := range values {
		delta := x - m
		m += delta / float64(i+1)
		s += delta * (x - m)
	}
	return math.Sqrt(s / n), m
}

// percentile returns the p-th percentile (0–100) of a pre-sorted slice
// using linear interpolation (matches R's quantile type=7 default).
func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return math.NaN()
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[n-1]
	}
	pos := (p / 100.0) * float64(n-1)
	lo := int(math.Floor(pos))
	hi := lo + 1
	if hi >= n {
		return sorted[n-1]
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}
