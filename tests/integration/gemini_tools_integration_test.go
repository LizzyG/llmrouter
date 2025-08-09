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
	ttools "github.com/lizzyg/llmrouter/tests/tools"
)

// Use shared test tools from tests/tools

func TestGemini_ToolWorkflow_LocationThenWeather(t *testing.T) {
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
	config.ResetForTest()

	client, err := llm.NewFromFile()
	if err != nil {
		t.Fatalf("NewFromFile: %v", err)
	}

	type Answer struct {
		Location string `json:"location"`
		Weather  string `json:"weather"`
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	prompt := "Use tools. First GetUserLocation, then GetWeatherInLocation with that location. Respond ONLY as JSON {location, weather}."
	got, err := llm.Execute[Answer](ctx, client, llm.Request{
		Model:       "gemini15pro",
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: prompt}},
		Tools:       ttools.LocationWeatherTools(),
		MaxTokens:   300,
		Temperature: 0.1,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got.Location != "Portland, Oregon" || got.Weather != "Sunny and mild in Portland, Oregon" {
		t.Fatalf("unexpected result: %+v", got)
	}
	fmt.Println("tool got", got)
}
