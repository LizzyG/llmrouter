package retry

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net"
	"time"
)

// Config holds retry configuration parameters
type Config struct {
	MaxAttempts int           `json:"max_attempts"`
	BaseDelay   time.Duration `json:"base_delay"`
	MaxDelay    time.Duration `json:"max_delay"`
	JitterRatio float64       `json:"jitter_ratio"`
}

// DefaultConfig returns the default retry configuration
func DefaultConfig() Config {
	return Config{
		MaxAttempts: 5,
		BaseDelay:   200 * time.Millisecond,
		MaxDelay:    3 * time.Second,
		JitterRatio: 0.25, // 25% jitter
	}
}

// WithRetry performs exponential backoff retries on transient errors.
func WithRetry(ctx context.Context, fn func() error) error {
	return WithRetryConfig(ctx, fn, DefaultConfig())
}

// WithRetryConfig performs exponential backoff retries with custom configuration.
func WithRetryConfig(ctx context.Context, fn func() error, config Config) error {
	var attempt int
	for {
		err := fn()
		if err == nil {
			return nil
		}
		if !IsTransient(err) {
			return err
		}
		attempt++
		if attempt >= config.MaxAttempts {
			return err
		}
		// Exponential backoff with jitter
		delay := time.Duration(float64(config.BaseDelay) * math.Pow(2, float64(attempt-1)))
		if delay > config.MaxDelay {
			delay = config.MaxDelay
		}
		// Add randomized jitter to prevent thundering herd
		jitter := time.Duration(rand.Float64() * config.JitterRatio * float64(delay))
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay + jitter):
		}
	}
}

// HTTPStatusError wraps HTTP status codes to enable reliable retry decisions.
type HTTPStatusError struct {
	Status int    `json:"status"`
	Body   string `json:"body"`
	Source string `json:"source"` // e.g., "openai", "gemini"
}

// NewHTTPStatusError creates a new HTTP status error
func NewHTTPStatusError(status int, body, source string) *HTTPStatusError {
	return &HTTPStatusError{
		Status: status,
		Body:   body,
		Source: source,
	}
}

func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf("%s http %d: %s", e.Source, e.Status, e.Body)
}

// IsTransient determines if an error is worth retrying using proper error type checking.
func IsTransient(err error) bool {
	// Retry on 429 or 5xx using proper error type
	var he *HTTPStatusError
	if errors.As(err, &he) {
		if he.Status == 429 || he.Status >= 500 {
			return true
		}
		return false
	}

	// Retry on network timeouts
	var ne net.Error
	if errors.As(err, &ne) {
		if ne.Timeout() {
			return true
		}
	}
	return false
}
