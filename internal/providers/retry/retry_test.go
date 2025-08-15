package retry

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()
	if config.MaxAttempts != 5 {
		t.Errorf("expected MaxAttempts 5, got %d", config.MaxAttempts)
	}
	if config.BaseDelay != 200*time.Millisecond {
		t.Errorf("expected BaseDelay 200ms, got %v", config.BaseDelay)
	}
	if config.MaxDelay != 3*time.Second {
		t.Errorf("expected MaxDelay 3s, got %v", config.MaxDelay)
	}
	if config.JitterRatio != 0.25 {
		t.Errorf("expected JitterRatio 0.25, got %f", config.JitterRatio)
	}
}

func TestWithRetryBehavior(t *testing.T) {
	t.Run("retry_with_transient_errors", func(t *testing.T) {
		callCount := 0
		start := time.Now()
		
		err := WithRetry(context.Background(), func() error {
			callCount++
			if callCount < 3 {
				// Return transient error for first 2 attempts
				return NewHTTPStatusError(429, "rate limited", "test")
			}
			// Succeed on 3rd attempt
			return nil
		})
		
		elapsed := time.Since(start)
		
		if err != nil {
			t.Fatalf("expected success after retries, got: %v", err)
		}
		if callCount != 3 {
			t.Fatalf("expected 3 calls (1 initial + 2 retries), got %d", callCount)
		}
		
		// Should have at least 2 delays: ~200ms + ~400ms = ~600ms minimum
		// With jitter, could be up to 25% more: ~750ms maximum
		minExpected := 500 * time.Millisecond
		maxExpected := 1000 * time.Millisecond // Extra buffer for test timing variance
		
		if elapsed < minExpected {
			t.Errorf("retry delays too short: expected at least %v, got %v", minExpected, elapsed)
		}
		if elapsed > maxExpected {
			t.Errorf("retry delays too long: expected at most %v, got %v", maxExpected, elapsed)
		}
	})
	
	t.Run("no_retry_on_non_transient_error", func(t *testing.T) {
		callCount := 0
		start := time.Now()
		
		err := WithRetry(context.Background(), func() error {
			callCount++
			// Return non-transient error
			return NewHTTPStatusError(400, "bad request", "test")
		})
		
		elapsed := time.Since(start)
		
		if err == nil {
			t.Fatal("expected error to be returned")
		}
		if callCount != 1 {
			t.Fatalf("expected 1 call (no retries), got %d", callCount)
		}
		
		// Should complete quickly with no delays
		maxExpected := 50 * time.Millisecond
		if elapsed > maxExpected {
			t.Errorf("non-transient error should not retry: expected at most %v, got %v", maxExpected, elapsed)
		}
	})
	
	t.Run("eventual_failure_after_max_attempts", func(t *testing.T) {
		callCount := 0
		
		err := WithRetry(context.Background(), func() error {
			callCount++
			// Always return transient error
			return NewHTTPStatusError(503, "service unavailable", "test")
		})
		
		if err == nil {
			t.Fatal("expected error after max attempts")
		}
		if callCount != 5 { // maxAttempts = 5
			t.Fatalf("expected 5 attempts, got %d", callCount)
		}
	})
	
	t.Run("context_cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		
		// Cancel immediately
		cancel()
		
		err := WithRetry(ctx, func() error {
			return NewHTTPStatusError(503, "service unavailable", "test")
		})
		
		if err != context.Canceled {
			t.Errorf("expected context.Canceled, got: %v", err)
		}
	})
}

func TestWithRetryConfig(t *testing.T) {
	config := Config{
		MaxAttempts: 2,
		BaseDelay:   50 * time.Millisecond,
		MaxDelay:    200 * time.Millisecond,
		JitterRatio: 0.1, // 10% jitter
	}
	
	callCount := 0
	start := time.Now()
	
	err := WithRetryConfig(context.Background(), func() error {
		callCount++
		if callCount < 2 {
			return NewHTTPStatusError(429, "rate limited", "test")
		}
		return nil
	}, config)
	
	elapsed := time.Since(start)
	
	if err != nil {
		t.Fatalf("expected success after retries, got: %v", err)
	}
	if callCount != 2 {
		t.Fatalf("expected 2 calls (1 initial + 1 retry), got %d", callCount)
	}
	
	// Should have 1 delay: ~50ms minimum, ~55ms with jitter
	minExpected := 40 * time.Millisecond
	maxExpected := 100 * time.Millisecond
	
	if elapsed < minExpected {
		t.Errorf("retry delays too short: expected at least %v, got %v", minExpected, elapsed)
	}
	if elapsed > maxExpected {
		t.Errorf("retry delays too long: expected at most %v, got %v", maxExpected, elapsed)
	}
}

func TestHTTPStatusError(t *testing.T) {
	err := NewHTTPStatusError(429, "rate limited", "openai")
	
	if err.Status != 429 {
		t.Errorf("expected status 429, got %d", err.Status)
	}
	if err.Body != "rate limited" {
		t.Errorf("expected body 'rate limited', got %s", err.Body)
	}
	if err.Source != "openai" {
		t.Errorf("expected source 'openai', got %s", err.Source)
	}
	
	expectedMsg := "openai http 429: rate limited"
	if err.Error() != expectedMsg {
		t.Errorf("expected error message '%s', got '%s'", expectedMsg, err.Error())
	}
}

func TestIsTransient(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "429 rate limit",
			err:      NewHTTPStatusError(429, "rate limited", "test"),
			expected: true,
		},
		{
			name:     "500 server error",
			err:      NewHTTPStatusError(500, "internal server error", "test"),
			expected: true,
		},
		{
			name:     "503 service unavailable",
			err:      NewHTTPStatusError(503, "service unavailable", "test"),
			expected: true,
		},
		{
			name:     "400 bad request",
			err:      NewHTTPStatusError(400, "bad request", "test"),
			expected: false,
		},
		{
			name:     "404 not found",
			err:      NewHTTPStatusError(404, "not found", "test"),
			expected: false,
		},
		{
			name:     "network timeout",
			err:      &net.DNSError{IsTimeout: true},
			expected: true,
		},
		{
			name:     "non-timeout network error",
			err:      &net.DNSError{IsTimeout: false},
			expected: false,
		},
		{
			name:     "generic error",
			err:      errors.New("generic error"),
			expected: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsTransient(tt.err)
			if result != tt.expected {
				t.Errorf("IsTransient(%v) = %v, expected %v", tt.err, result, tt.expected)
			}
		})
	}
}
