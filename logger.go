// Package logging provides structured, leveled logging for the bootstrap application.
// Log entries are written to both stdout and a log file for auditing and diagnostics.
package logging

import (
	"fmt"
	"io"
	"log"
	"os"
	"time"
)

// Level represents the severity of a log message.
type Level int

const (
	DEBUG Level = iota
	INFO
	WARN
	ERROR
)

func (l Level) String() string {
	switch l {
	case DEBUG:
		return "DEBUG"
	case INFO:
		return "INFO"
	case WARN:
		return "WARN"
	case ERROR:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Logger writes structured log entries to one or more writers.
type Logger struct {
	minLevel Level
	inner    *log.Logger
}

// New creates a Logger that writes to all provided writers at minLevel and above.
func New(minLevel Level, writers ...io.Writer) *Logger {
	multi := io.MultiWriter(writers...)
	return &Logger{
		minLevel: minLevel,
		inner:    log.New(multi, "", 0),
	}
}

// NewFileLogger opens (or creates) logPath and returns a Logger writing to both
// that file and stdout. The caller is responsible for closing the returned *os.File.
func NewFileLogger(logPath string, minLevel Level) (*Logger, *os.File, error) {
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, nil, fmt.Errorf("logging: cannot open log file %q: %w", logPath, err)
	}
	return New(minLevel, os.Stdout, f), f, nil
}

func (l *Logger) log(level Level, format string, args ...any) {
	if level < l.minLevel {
		return
	}
	ts := time.Now().Format("2006-01-02 15:04:05.000")
	msg := fmt.Sprintf(format, args...)
	l.inner.Printf("[%s] [%s] %s", ts, level, msg)
}

// Debug logs a message at DEBUG level.
func (l *Logger) Debug(format string, args ...any) { l.log(DEBUG, format, args...) }

// Info logs a message at INFO level.
func (l *Logger) Info(format string, args ...any) { l.log(INFO, format, args...) }

// Warn logs a message at WARN level.
func (l *Logger) Warn(format string, args ...any) { l.log(WARN, format, args...) }

// Error logs a message at ERROR level.
func (l *Logger) Error(format string, args ...any) { l.log(ERROR, format, args...) }
