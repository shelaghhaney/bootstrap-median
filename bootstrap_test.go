package bootstrap

import (
	"math"
	"testing"
)

// ─── MT19937 Tests ───────────────────────────────────────────────────────────

// TestMT19937Reproducibility verifies that the same seed produces the same sequence
// on repeated construction — the essential property for reproducible research.
func TestMT19937Reproducibility(t *testing.T) {
	const seed uint32 = 19650218
	const draws = 20
	rng1 := NewMT19937(seed)
	rng2 := NewMT19937(seed)
	for i := 0; i < draws; i++ {
		v1, v2 := rng1.Uint32(), rng2.Uint32()
		if v1 != v2 {
			t.Errorf("draw %d: rng1=%d != rng2=%d (same seed should produce same output)", i, v1, v2)
		}
	}
}

// TestMT19937DifferentSeeds verifies that two different seeds yield different sequences.
func TestMT19937DifferentSeeds(t *testing.T) {
	rng1 := NewMT19937(1)
	rng2 := NewMT19937(2)
	same := 0
	for i := 0; i < 100; i++ {
		if rng1.Uint32() == rng2.Uint32() {
			same++
		}
	}
	if same > 5 {
		t.Errorf("Different seeds produced %d/100 identical values (expected near 0)", same)
	}
}

// TestMT19937Float64Range verifies Float64 output lies in [0,1).
func TestMT19937Float64Range(t *testing.T) {
	rng := NewMT19937(42)
	for i := 0; i < 100_000; i++ {
		v := rng.Float64()
		if v < 0 || v >= 1 {
			t.Fatalf("Float64 out of range [0,1): got %v at iteration %d", v, i)
		}
	}
}

// TestMT19937IntnDistribution checks that Intn produces a roughly uniform
// distribution over [0, k) using a chi-squared-like check.
func TestMT19937IntnDistribution(t *testing.T) {
	const k = 10
	const draws = 100_000
	rng := NewMT19937(7)
	counts := make([]int, k)
	for i := 0; i < draws; i++ {
		counts[rng.Intn(k)]++
	}
	expected := float64(draws) / k
	for i, c := range counts {
		ratio := float64(c) / expected
		if ratio < 0.95 || ratio > 1.05 {
			t.Errorf("Intn bucket %d: count=%d, expected≈%.0f (ratio=%.3f)", i, c, expected, ratio)
		}
	}
}

// ─── Estimator Tests ─────────────────────────────────────────────────────────

func TestMedianOdd(t *testing.T) {
	data := []float64{3, 1, 4, 1, 5}
	got := Median(data)
	want := 3.0
	if got != want {
		t.Errorf("Median(odd): got %v, want %v", got, want)
	}
}

func TestMedianEven(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	got := Median(data)
	want := 2.5
	if got != want {
		t.Errorf("Median(even): got %v, want %v", got, want)
	}
}

func TestMedianSingle(t *testing.T) {
	data := []float64{7.0}
	got := Median(data)
	if got != 7.0 {
		t.Errorf("Median(single): got %v, want 7.0", got)
	}
}

func TestMeanBasic(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	got := Mean(data)
	want := 3.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Mean: got %v, want %v", got, want)
	}
}

// ─── Percentile Tests ─────────────────────────────────────────────────────────

func TestPercentile(t *testing.T) {
	sorted := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	tests := []struct {
		p    float64
		want float64
	}{
		{0, 1},
		{100, 10},
		{50, 5.5}, // linear interp between 5 and 6
	}
	for _, tt := range tests {
		got := percentile(sorted, tt.p)
		if math.Abs(got-tt.want) > 1e-9 {
			t.Errorf("percentile(%v): got %v, want %v", tt.p, got, tt.want)
		}
	}
}

// ─── Bootstrap Integration Tests ─────────────────────────────────────────────

// TestRunNormalSE verifies that for a Normal(0,1) sample the bootstrap SE of
// the median is close to the theoretical value ≈ sqrt(π/2)/sqrt(n).
// For n=1000, theory gives ≈ 0.0396.
func TestRunNormalSE(t *testing.T) {
	rng := NewMT19937(42)
	data := GenerateNormal(rng, 1000, 0, 1)
	cfg := Config{NumResamples: 2000, Seed: 99}

	result, err := Run(data, Median, cfg)
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	// Theoretical SE of the median for N(0,1), n=1000: sqrt(π/2)/sqrt(1000) ≈ 0.0396
	theoreticalSE := math.Sqrt(math.Pi/2.0) / math.Sqrt(1000)
	tolerance := 0.015 // allow ±15% for Monte Carlo noise at B=2000

	if math.Abs(result.BootstrapSE-theoreticalSE) > tolerance {
		t.Errorf("Bootstrap SE = %.4f; theoretical = %.4f; diff = %.4f (tol=%.4f)",
			result.BootstrapSE, theoreticalSE,
			math.Abs(result.BootstrapSE-theoreticalSE), tolerance)
	}
}

// TestRunInvalidInput verifies that Run returns errors for bad inputs.
func TestRunInvalidInput(t *testing.T) {
	cfg := DefaultConfig()

	_, err := Run([]float64{}, Median, cfg)
	if err == nil {
		t.Error("Expected error for empty data, got nil")
	}

	_, err = Run([]float64{1.0}, Median, cfg)
	if err == nil {
		t.Error("Expected error for single-element data, got nil")
	}

	cfg2 := Config{NumResamples: 0, Seed: 1}
	_, err = Run([]float64{1.0, 2.0}, Median, cfg2)
	if err == nil {
		t.Error("Expected error for NumResamples=0, got nil")
	}
}

// TestRunCI95Coverage checks that the 95% CI contains the true median (0 for
// Normal(0,1)) in a high fraction of replications – a basic coverage test.
func TestRunCI95Coverage(t *testing.T) {
	const reps = 200
	const n = 200
	hits := 0
	for i := 0; i < reps; i++ {
		rng := NewMT19937(uint32(i * 1337))
		data := GenerateNormal(rng, n, 0, 1)
		cfg := Config{NumResamples: 500, Seed: uint32(i)}
		result, err := Run(data, Median, cfg)
		if err != nil {
			t.Fatalf("Run error on rep %d: %v", i, err)
		}
		if result.CI95Lower <= 0 && 0 <= result.CI95Upper {
			hits++
		}
	}
	coverage := float64(hits) / reps
	if coverage < 0.88 { // expect ≥88% (nominal 95%, reduced for small B)
		t.Errorf("CI95 coverage = %.1f%%; expected >= 88%%", coverage*100)
	}
}

// ─── Data Generation Tests ────────────────────────────────────────────────────

func TestGenerateNormalLength(t *testing.T) {
	rng := NewMT19937(1)
	data := GenerateNormal(rng, 101, 0, 1)
	if len(data) != 101 {
		t.Errorf("GenerateNormal length: got %d, want 101", len(data))
	}
}

func TestGenerateLogNormalPositive(t *testing.T) {
	rng := NewMT19937(2)
	data := GenerateLogNormal(rng, 500, 0, 1)
	for i, v := range data {
		if v <= 0 {
			t.Errorf("LogNormal[%d] = %v; expected positive", i, v)
		}
	}
}

func TestGenerateNegativelySkewed(t *testing.T) {
	rng := NewMT19937(3)
	data := GenerateNegativelySkewed(rng, 500, 0, 1)
	// mean < median for negatively skewed
	m := Mean(data)
	med := Median(data)
	if m >= med {
		t.Errorf("Negatively skewed: mean (%.4f) should be < median (%.4f)", m, med)
	}
}

// ─── Benchmarks ───────────────────────────────────────────────────────────────

// BenchmarkBootstrapMedian_n500_B1000 benchmarks the full bootstrap pipeline
// for n=500 observations and B=1000 resamples.
func BenchmarkBootstrapMedian_n500_B1000(b *testing.B) {
	rng := NewMT19937(42)
	data := GenerateNormal(rng, 500, 0, 1)
	cfg := Config{NumResamples: 1000, Seed: 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Run(data, Median, cfg)
	}
}

// BenchmarkBootstrapMedian_n5000_B1000 scales up to n=5000.
func BenchmarkBootstrapMedian_n5000_B1000(b *testing.B) {
	rng := NewMT19937(42)
	data := GenerateNormal(rng, 5000, 0, 1)
	cfg := Config{NumResamples: 1000, Seed: 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Run(data, Median, cfg)
	}
}

// BenchmarkMedian_n500 benchmarks the Median function alone.
func BenchmarkMedian_n500(b *testing.B) {
	rng := NewMT19937(42)
	data := GenerateNormal(rng, 500, 0, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Median(data)
	}
}

// BenchmarkMT19937_Uint32 benchmarks raw PRNG throughput.
func BenchmarkMT19937_Uint32(b *testing.B) {
	rng := NewMT19937(1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rng.Uint32()
	}
}
