package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/lizzyg/llmrouter/internal/config"
	"github.com/lizzyg/llmrouter/internal/core"
)

// This is a minimal smoke test ensuring the client can be constructed.
// Network calls are not made here.
func TestNewClient(t *testing.T) {
    c := New(config.ModelConfig{APIKey: "test", Model: "gpt-4o"}, &http.Client{}, nil)
    if c == nil {
        t.Fatal("expected client")
    }
}

func TestMapChatMessages_StructuredToolResults(t *testing.T) {
	// Test that structured ToolResults are properly mapped with JSON marshaling
	msgs := []core.Message{{
		Role: "assistant",
		ToolResults: []core.ToolResult{{
			CallID: "abc123",
			Name:   "Weather",
			Result: map[string]any{"temp": 72, "condition": "sunny"},
		}},
	}}
	mapped := mapChatMessages(msgs)
	if len(mapped) != 1 {
		t.Fatalf("expected 1 mapped message, got %d", len(mapped))
	}
	m := mapped[0]
	if m["role"] != "tool" {
		t.Fatalf("expected role tool, got %v", m["role"])
	}
	if m["name"] != "Weather" {
		t.Fatalf("expected name Weather, got %v", m["name"])
	}
	if m["tool_call_id"] != "abc123" {
		t.Fatalf("expected tool_call_id abc123, got %v", m["tool_call_id"])
	}
	content, ok := m["content"].(string)
	if !ok || content == "" {
		t.Fatalf("expected non-empty string content, got %v", m["content"])
	}
	// Verify it's valid JSON
	var result map[string]any
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		t.Fatalf("content should be valid JSON, got: %s, error: %v", content, err)
	}
	if result["temp"].(float64) != 72 {
		t.Fatalf("expected temp 72, got %v", result["temp"])
	}
}

func TestMapChatMessages_UnmarshalableToolResult(t *testing.T) {
	// Test that unmarshalable tool results produce valid JSON error objects
	ch := make(chan int) // channels can't be marshaled
	msgs := []core.Message{{
		Role: "assistant",
		ToolResults: []core.ToolResult{{
			CallID: "test123",
			Name:   "InvalidTool",
			Result: ch,
		}},
	}}
	mapped := mapChatMessages(msgs)
	if len(mapped) != 1 {
		t.Fatalf("expected 1 mapped message, got %d", len(mapped))
	}
	m := mapped[0]
	if m["role"] != "tool" {
		t.Fatalf("expected role tool, got %v", m["role"])
	}
	content, ok := m["content"].(string)
	if !ok || content == "" {
		t.Fatalf("expected non-empty string content, got %v", m["content"])
	}
	// Verify it's valid JSON
	var errorResult map[string]any
	if err := json.Unmarshal([]byte(content), &errorResult); err != nil {
		t.Fatalf("error content should be valid JSON, got: %s, error: %v", content, err)
	}
	// Verify it contains the error message
	errorMsg, ok := errorResult["error"].(string)
	if !ok {
		t.Fatalf("expected error field to be string, got %v", errorResult["error"])
	}
	if errorMsg == "" {
		t.Fatalf("expected non-empty error message")
	}
}

func TestWithRetryBehavior(t *testing.T) {
	// Test actual retry behavior with timing verification
	c := &Client{}
	
	t.Run("retry_with_transient_errors", func(t *testing.T) {
		callCount := 0
		start := time.Now()
		
		err := c.withRetry(context.Background(), func() error {
			callCount++
			if callCount < 3 {
				// Return transient error for first 2 attempts
				return &httpStatusError{status: 429, body: "rate limited"}
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
		
		err := c.withRetry(context.Background(), func() error {
			callCount++
			// Return non-transient error
			return &httpStatusError{status: 400, body: "bad request"}
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
		
		err := c.withRetry(context.Background(), func() error {
			callCount++
			// Always return transient error
			return &httpStatusError{status: 503, body: "service unavailable"}
		})
		
		if err == nil {
			t.Fatal("expected error after max attempts")
		}
		if callCount != 5 { // maxAttempts = 5
			t.Fatalf("expected 5 attempts, got %d", callCount)
		}
	})
}
