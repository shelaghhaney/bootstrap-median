package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTemp writes content to a temporary file and returns its path.
func writeTemp(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "test_*.csv")
	if err != nil {
		t.Fatalf("writeTemp: %v", err)
	}
	if _, err := f.WriteString(content); err != nil {
		t.Fatalf("writeTemp write: %v", err)
	}
	f.Close()
	return f.Name()
}

// TestLoadCSV_ValidFile checks that a well-formed CSV (as written by R's
// write.csv) is parsed correctly.
func TestLoadCSV_ValidFile(t *testing.T) {
	// R writes: header row, then "value" column
	content := "\"value\"\n1.5\n2.5\n3.0\n-1.2\n0.0\n"
	path := writeTemp(t, content)

	values, err := loadCSV(path)
	if err != nil {
		t.Fatalf("loadCSV returned error: %v", err)
	}
	if len(values) != 5 {
		t.Errorf("expected 5 values, got %d", len(values))
	}
	if values[0] != 1.5 {
		t.Errorf("values[0]: got %v, want 1.5", values[0])
	}
	if values[3] != -1.2 {
		t.Errorf("values[3]: got %v, want -1.2", values[3])
	}
}

// TestLoadCSV_TooFewRows verifies that a file with only one data row is rejected.
func TestLoadCSV_TooFewRows(t *testing.T) {
	content := "\"value\"\n42.0\n"
	path := writeTemp(t, content)
	_, err := loadCSV(path)
	if err == nil {
		t.Error("expected error for single-row file, got nil")
	}
}

// TestLoadCSV_NonNumeric verifies that a non-numeric value triggers an error.
func TestLoadCSV_NonNumeric(t *testing.T) {
	content := "\"value\"\n1.0\nnot_a_number\n3.0\n"
	path := writeTemp(t, content)
	_, err := loadCSV(path)
	if err == nil {
		t.Error("expected error for non-numeric value, got nil")
	}
}

// TestLoadCSV_MissingFile verifies that a missing file returns an error.
func TestLoadCSV_MissingFile(t *testing.T) {
	_, err := loadCSV(filepath.Join(t.TempDir(), "does_not_exist.csv"))
	if err == nil {
		t.Error("expected error for missing file, got nil")
	}
}

// TestLoadCSV_LargeFile verifies that a 500-row file (matching the assignment
// minimum of 100 observations) is loaded completely.
func TestLoadCSV_LargeFile(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("\"value\"\n")
	for i := 0; i < 500; i++ {
		sb.WriteString("1.0\n")
	}
	path := writeTemp(t, sb.String())

	values, err := loadCSV(path)
	if err != nil {
		t.Fatalf("loadCSV error on large file: %v", err)
	}
	if len(values) != 500 {
		t.Errorf("expected 500 values, got %d", len(values))
	}
}

// TestSaveSummaryCSV_RoundTrip writes a summary CSV and verifies it is non-empty.
func TestSaveSummaryCSV_RoundTrip(t *testing.T) {
	records := []runRecord{
		{Shape: "Symmetric (Normal)", N: 500, B: 1000,
			Estimate: 0.01, SE: 0.05, Bias: -0.001,
			CI95Lo: -0.09, CI95Hi: 0.09, ElapsedMs: 42.1, AllocKB: 4000},
	}
	path := filepath.Join(t.TempDir(), "summary.csv")
	if err := saveSummaryCSV(path, records); err != nil {
		t.Fatalf("saveSummaryCSV error: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat error: %v", err)
	}
	if info.Size() == 0 {
		t.Error("summary CSV is empty")
	}
}
