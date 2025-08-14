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

func TestMapChatMessages_ToolResultMapping(t *testing.T) {
    // Router encodes tool results as an array of {tool, result, tool_call_id}
    results := []map[string]any{{
        "tool":         "Weather",
        "result":       map[string]any{"temp": 72},
        "tool_call_id": "abc123",
    }}
    b, _ := json.Marshal(results)
    msgs := []core.Message{{
        Role:    "assistant",
        Content: string(b),
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
