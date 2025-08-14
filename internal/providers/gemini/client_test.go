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
