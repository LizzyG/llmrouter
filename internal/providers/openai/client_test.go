package openai

import (
	"encoding/json"
	"net/http"
	"testing"

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
