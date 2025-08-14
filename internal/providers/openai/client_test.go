package openai

import (
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
	// Test that structured ToolResults are properly mapped
	msgs := []core.Message{{
		Role: "assistant",
		ToolResults: []core.ToolResult{{
			CallID: "abc123",
			Name:   "Weather",
			Result: map[string]any{"temp": 72},
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
	if m["content"] == "" {
		t.Fatalf("expected non-empty content")
	}
}
