//go:build integration
// +build integration

package integration

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	_ "github.com/joho/godotenv/autoload"
	llm "github.com/lizzyg/llmrouter"
	"github.com/lizzyg/llmrouter/internal/config"
)

func TestOpenAI_Execute_TypedJSON(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set; skipping integration test")
	}

	// Write a temporary config that embeds the API key directly.
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(cfgPath, []byte(`llm:
  models:
    gpt4o:
      provider: openai
      model: gpt-4o
      api_key: `+apiKey+`
      supports_web_search: false
      supports_tools: true
      supports_structured_output: true
      context_window: 128000
      max_output_tokens: 400
`), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	t.Setenv("LLM_CONFIG_PATH", cfgPath)
	config.ResetForTest()

	client, err := llm.NewFromFile()
	if err != nil {
		t.Fatalf("NewFromFile: %v", err)
	}

	type Answer struct {
		Headline string   `json:"headline"`
		Points   []string `json:"points"`
		Stats    struct {
			Count int     `json:"count"`
			Score float64 `json:"score"`
		} `json:"stats"`
		Success bool `json:"success"`
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	got, err := llm.Execute[Answer](ctx, client, llm.Request{
		Model: "gpt4o",
		Messages: []llm.Message{{
			Role:    llm.RoleUser,
			Content: "Respond ONLY as JSON matching this shape: {headline: string, points: string[3], stats: {count: integer > 0, score: number 0..1}, success: boolean}. Summarize Go 1.22 features.",
		}},
		MaxTokens:   200,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got.Headline == "" || len(got.Points) != 3 || got.Stats.Count <= 0 || got.Stats.Score < 0 || got.Stats.Score > 1 {
		t.Fatalf("unexpected structured result: %+v", got)
	}
	fmt.Println("got", got)
}
