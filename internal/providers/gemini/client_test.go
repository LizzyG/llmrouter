package gemini

import (
	"net/http"
	"testing"

	"github.com/lizzyg/llmrouter/internal/config"
)

func TestNewClient(t *testing.T) {
	c := New(config.ModelConfig{APIKey: "test", Model: "gemini-1.5-pro"}, &http.Client{}, nil)
	if c == nil {
		t.Fatal("expected client")
	}
}
