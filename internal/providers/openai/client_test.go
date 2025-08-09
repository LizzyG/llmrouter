package openai

import (
	"net/http"
	"testing"

	"github.com/lizzyg/llmrouter/internal/config"
)

// This is a minimal smoke test ensuring the client can be constructed.
// Network calls are not made here.
func TestNewClient(t *testing.T) {
	c := New(config.ModelConfig{APIKey: "test", Model: "gpt-4o"}, &http.Client{}, nil)
	if c == nil {
		t.Fatal("expected client")
	}
}
