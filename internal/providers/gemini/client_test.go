package gemini

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"testing"
	"time"

	"github.com/lizzyg/llmrouter/internal/config"
	"github.com/lizzyg/llmrouter/internal/core"
	"github.com/lizzyg/llmrouter/internal/providers/retry"
)

func TestNewClient(t *testing.T) {
	c := New(config.ModelConfig{APIKey: "test", Model: "gemini-1.5-pro"}, &http.Client{}, nil)
	if c == nil {
		t.Fatal("expected client")
	}
}

func TestIsTransient(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{"nil error", nil, false},
		{"429 status", NewHTTPStatusError(429, "rate limited"), true},
		{"500 status", NewHTTPStatusError(500, "server error"), true},
		{"503 status", NewHTTPStatusError(503, "service unavailable"), true},
		{"400 status", NewHTTPStatusError(400, "bad request"), false},
		{"401 status", NewHTTPStatusError(401, "unauthorized"), false},
		{"404 status", NewHTTPStatusError(404, "not found"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := retry.IsTransient(tt.err)
			if got != tt.expected {
				t.Errorf("isTransient(%v) = %v, want %v", tt.err, got, tt.expected)
			}
		})
	}
}

func TestWithRetryBehavior(t *testing.T) {
	// Test actual retry behavior with timing verification
	
	t.Run("retry_with_transient_errors", func(t *testing.T) {
		callCount := 0
		start := time.Now()
		
		err := withRetry(context.Background(), func() error {
			callCount++
			if callCount < 3 {
				// Return transient error for first 2 attempts
				return NewHTTPStatusError(429, "rate limited")
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
		
		err := withRetry(context.Background(), func() error {
			callCount++
			// Return non-transient error
			return NewHTTPStatusError(400, "bad request")
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
		
		err := withRetry(context.Background(), func() error {
			callCount++
			// Always return transient error
			return NewHTTPStatusError(503, "service unavailable")
		})
		
		if err == nil {
			t.Fatal("expected error after max attempts")
		}
		if callCount != 5 { // maxAttempts = 5
			t.Fatalf("expected 5 attempts, got %d", callCount)
		}
	})
}

func TestMapMessages_InvalidToolCallArgs(t *testing.T) {
	// Test that invalid JSON in tool call args is handled gracefully
	c := &Client{
		logger: testLogger(),
	}
	
	msgs := []core.Message{{
		Role: "assistant",
		ToolCalls: []core.ToolCall{{
			Name: "TestTool",
			Args: []byte(`{invalid json`), // Invalid JSON
		}},
	}}
	
	// Should not panic and should log the error
	result := mapMessages(msgs, c.logger)
	
	// Verify the message was still processed
	if len(result) != 1 {
		t.Fatalf("expected 1 mapped message, got %d", len(result))
	}
	
	msg := result[0]
	if msg["role"] != "model" {
		t.Fatalf("expected role model, got %v", msg["role"])
	}
	
	// Verify parts contain the function call with nil args
	parts, ok := msg["parts"].([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("expected 1 part, got %v", msg["parts"])
	}
	
	part := parts[0].(map[string]any)
	funcCall, ok := part["functionCall"].(map[string]any)
	if !ok {
		t.Fatalf("expected functionCall in part")
	}
	
	if funcCall["name"] != "TestTool" {
		t.Fatalf("expected tool name TestTool, got %v", funcCall["name"])
	}
	
	// args should be nil due to unmarshal failure
	if funcCall["args"] != nil {
		t.Fatalf("expected nil args due to unmarshal error, got %v", funcCall["args"])
	}
}

// testLogger returns a logger that discards output for testing
func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
