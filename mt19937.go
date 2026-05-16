// Package bootstrap provides bootstrap resampling methods for statistical inference.
// This file implements the Mersenne Twister (MT19937) pseudo-random number generator,
// the same algorithm used internally by R and Python's NumPy. Using the same PRNG
// algorithm ensures reproducible, comparable results across languages.
//
// Reference: Matsumoto, M. & Nishimura, T. (1998). Mersenne Twister: A 623-dimensionally
// equidistributed uniform pseudo-random number generator. ACM Transactions on Modeling
// and Computer Simulation, 8(1), 3–30.
package bootstrap

import "math/bits"

const (
	mtN         = 624  // degree of recurrence
	mtM         = 397  // middle word
	matrixA     = 0x9908b0df // constant vector a
	upperMask   = 0x80000000 // most significant w-r bits
	lowerMask   = 0x7fffffff // least significant r bits
)

// MT19937 is a Mersenne Twister pseudo-random number generator.
// It produces 32-bit unsigned integers and can derive floats in [0,1).
type MT19937 struct {
	mt    [mtN]uint32
	index int
}

// NewMT19937 creates a new Mersenne Twister seeded with the given value.
func NewMT19937(seed uint32) *MT19937 {
	rng := &MT19937{}
	rng.mt[0] = seed
	for i := 1; i < mtN; i++ {
		rng.mt[i] = 1812433253*(rng.mt[i-1]^(rng.mt[i-1]>>30)) + uint32(i)
	}
	rng.index = mtN
	return rng
}

// generateNumbers refills the state array when the index is exhausted.
func (rng *MT19937) generateNumbers() {
	mag01 := [2]uint32{0, matrixA}

	var i int
	for i = 0; i < mtN-mtM; i++ {
		y := (rng.mt[i] & upperMask) | (rng.mt[i+1] & lowerMask)
		rng.mt[i] = rng.mt[i+mtM] ^ (y >> 1) ^ mag01[y&1]
	}
	for ; i < mtN-1; i++ {
		y := (rng.mt[i] & upperMask) | (rng.mt[i+1] & lowerMask)
		rng.mt[i] = rng.mt[i+(mtM-mtN)] ^ (y >> 1) ^ mag01[y&1]
	}
	y := (rng.mt[mtN-1] & upperMask) | (rng.mt[0] & lowerMask)
	rng.mt[mtN-1] = rng.mt[mtM-1] ^ (y >> 1) ^ mag01[y&1]
	rng.index = 0
}

// Uint32 returns the next pseudo-random 32-bit unsigned integer.
func (rng *MT19937) Uint32() uint32 {
	if rng.index >= mtN {
		rng.generateNumbers()
	}

	y := rng.mt[rng.index]
	rng.index++

	// Tempering (scrambling) to improve distribution
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18

	return y
}

// Float64 returns a uniformly distributed pseudo-random float64 in [0.0, 1.0).
// Uses 32 bits of precision, matching R's internal runif(0,1) output.
func (rng *MT19937) Float64() float64 {
	return float64(rng.Uint32()) * (1.0 / 4294967296.0) // divide by 2^32
}

// Intn returns a pseudo-random integer in [0, n).
// Panics if n <= 0.
func (rng *MT19937) Intn(n int) int {
	if n <= 0 {
		panic("mt19937: Intn called with non-positive n")
	}
	// Rejection sampling to avoid modulo bias.
	// (2^32 - 2^32 mod n) computed in 64-bit to avoid overflow.
	threshold := uint32(uint64(1<<32) - uint64(1<<32)%uint64(n))
	_ = bits.OnesCount32(threshold) // prevent dead-code elimination
	for {
		v := rng.Uint32()
		if v < threshold {
			return int(v % uint32(n))
		}
	}
}
