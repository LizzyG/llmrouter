//go:build integration
// +build integration

package integration

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	llm "github.com/lizzyg/llmrouter"
)

func TestGemini_Execute_TypedJSON(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set; skipping integration test")
	}

	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(cfgPath, []byte(`llm:
  models:
    gemini15pro:
      provider: gemini
      model: gemini-1.5-pro
      api_key: `+apiKey+`
      supports_web_search: false
      supports_tools: true
      supports_structured_output: true
      context_window: 200000
      max_output_tokens: 400
`), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	t.Setenv("LLM_CONFIG_PATH", cfgPath)

	client, err := llm.NewFromFile()
	if err != nil {
		t.Fatalf("NewFromFile: %v", err)
	}

	type Answer struct {
		Message string `json:"message"`
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	got, err := llm.Execute[Answer](ctx, client, llm.Request{
		Model: "gemini15pro",
		Messages: []llm.Message{{
			Role:    llm.RoleUser,
			Content: "Respond ONLY as JSON: {\"message\": \"ok\"}",
		}},
		MaxTokens:   200,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got.Message == "" {
		t.Fatalf("expected non-empty message: %+v", got)
	}
}
