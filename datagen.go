// Package bootstrap - data generation utilities.
// Generates synthetic datasets with controllable distributional shapes
// (normal/symmetric, positively skewed, negatively skewed) using MT19937,
// matching the shapes used in the corresponding R analysis.
package bootstrap

import "math"

// DistributionShape describes the skewness character of generated data.
type DistributionShape string

const (
	ShapeSymmetric        DistributionShape = "symmetric"        // Normal(0,1)
	ShapePositivelySkewed DistributionShape = "positively_skewed" // Log-normal
	ShapeNegativelySkewed DistributionShape = "negatively_skewed" // Reflected log-normal
)

// GenerateNormal produces n observations from Normal(mean, sd) using the
// Box-Muller transform applied to MT19937 uniform variates.
// This matches the distributional family used in R's rnorm().
func GenerateNormal(rng *MT19937, n int, mean, sd float64) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i += 2 {
		u1 := rng.Float64()
		u2 := rng.Float64()
		// Box-Muller transform
		r := math.Sqrt(-2.0 * math.Log(u1+1e-300)) // guard against log(0)
		theta := 2.0 * math.Pi * u2
		z0 := r * math.Cos(theta)
		z1 := r * math.Sin(theta)
		out[i] = mean + sd*z0
		if i+1 < n {
			out[i+1] = mean + sd*z1
		}
	}
	return out
}

// GenerateLogNormal produces n observations from LogNormal(mu, sigma).
// The resulting distribution is positively (right) skewed.
func GenerateLogNormal(rng *MT19937, n int, mu, sigma float64) []float64 {
	normals := GenerateNormal(rng, n, mu, sigma)
	out := make([]float64, n)
	for i, v := range normals {
		out[i] = math.Exp(v)
	}
	return out
}

// GenerateNegativelySkewed produces n observations from a distribution that is
// negatively (left) skewed by reflecting a log-normal around its midpoint.
func GenerateNegativelySkewed(rng *MT19937, n int, mu, sigma float64) []float64 {
	raw := GenerateLogNormal(rng, n, mu, sigma)
	// Reflect: x' = max - x  →  left-skewed mirror image
	var maxVal float64
	for _, v := range raw {
		if v > maxVal {
			maxVal = v
		}
	}
	out := make([]float64, n)
	for i, v := range raw {
		out[i] = maxVal - v
	}
	return out
}

// GenerateDataset returns a named dataset for one of the three shape types.
func GenerateDataset(rng *MT19937, shape DistributionShape, n int) []float64 {
	switch shape {
	case ShapeSymmetric:
		return GenerateNormal(rng, n, 0, 1)
	case ShapePositivelySkewed:
		return GenerateLogNormal(rng, n, 0, 1)
	case ShapeNegativelySkewed:
		return GenerateNegativelySkewed(rng, n, 0, 1)
	default:
		return GenerateNormal(rng, n, 0, 1)
	}
}
