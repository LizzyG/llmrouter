package integration

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	_ "github.com/joho/godotenv/autoload"
	llm "github.com/lizzyg/llmrouter"
	"github.com/lizzyg/llmrouter/internal/config"
	ttools "github.com/lizzyg/llmrouter/tests/tools"
)

// Use shared test tools from tests/tools

func TestGemini_ToolWorkflow_LocationThenWeather(t *testing.T) {
	os.Setenv("LLM_VERBOSE_MESSAGES", "1")
	os.Setenv("LLM_CONFIG_PATH", "../../config.yaml")
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set; skipping integration test")
	}
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

	prompt := "Use the available tools to answer. First call GetUserLocation (no args) to get the user's location.  Once you have the location, then call GetWeatherInLocation with {location} from the first tool. Finally, respond ONLY as JSON with fields: location, weather."
	got, err := llm.Execute[Answer](ctx, client, llm.Request{
		Model:       "gemini15flash",
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: prompt}},
		Tools:       ttools.LocationWeatherTools(),
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
