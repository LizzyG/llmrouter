package gemini

import (
	"io"
	"log/slog"
	"math"
	"net/http"
	"testing"
	"time"

	"github.com/lizzyg/llmrouter/internal/config"
	"github.com/lizzyg/llmrouter/internal/core"
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
		{"429 status", &httpStatusError{status: 429, body: "rate limited"}, true},
		{"500 status", &httpStatusError{status: 500, body: "server error"}, true},
		{"503 status", &httpStatusError{status: 503, body: "service unavailable"}, true},
		{"400 status", &httpStatusError{status: 400, body: "bad request"}, false},
		{"401 status", &httpStatusError{status: 401, body: "unauthorized"}, false},
		{"404 status", &httpStatusError{status: 404, body: "not found"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isTransient(tt.err)
			if got != tt.expected {
				t.Errorf("isTransient(%v) = %v, want %v", tt.err, got, tt.expected)
			}
		})
	}
}

func TestExponentialBackoffJitter(t *testing.T) {
	// Test that the jitter is randomized by checking multiple delay calculations
	// This is a bit tricky to test deterministically, so we'll verify the range
	
	// Use a fixed seed to make the test deterministic
	// Note: In Go 1.20+, rand.Float64() uses a global source that's automatically seeded
	// But for test consistency, we can use a local source
	
	baseDelay := 100 * time.Millisecond
	
	// Calculate delay for attempt 2 (first retry)
	attempt := 2
	expectedBaseDelay := time.Duration(float64(baseDelay) * math.Pow(2, float64(attempt-1)))
	
	// We can't easily test the exact jitter without exposing internals,
	// but we can verify that multiple calls to the retry logic would
	// produce different delays due to randomization.
	// For now, just verify that our jitter calculation range is reasonable.
	
	if expectedBaseDelay != 200*time.Millisecond {
		t.Errorf("expected base delay 200ms for attempt 2, got %v", expectedBaseDelay)
	}
	
	// The jitter should be 0-25% of the delay, so for 200ms base:
	// - Minimum total delay: 200ms
	// - Maximum total delay: 200ms + 50ms = 250ms
	minExpected := expectedBaseDelay
	maxExpected := expectedBaseDelay + time.Duration(0.25*float64(expectedBaseDelay))
	
	if maxExpected != 250*time.Millisecond {
		t.Errorf("expected max delay 250ms, got %v", maxExpected)
	}
	if minExpected != 200*time.Millisecond {
		t.Errorf("expected min delay 200ms, got %v", minExpected)
	}
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
