package llmrouter

import (
	"context"
	"errors"
	"log/slog"
	"testing"

	moderr "github.com/lizzyg/llmrouter/errors"
	"github.com/lizzyg/llmrouter/internal/config"
	"github.com/lizzyg/llmrouter/internal/core"
)

type fakeClient struct {
	calls     int
	lastModel string
	responses []RawResponse
}

func (f *fakeClient) Call(ctx context.Context, p CallParams) (RawResponse, error) {
	f.calls++
	f.lastModel = p.Model
	if len(f.responses) == 0 {
		return RawResponse{}, nil
	}
	r := f.responses[0]
	f.responses = f.responses[1:]
	return r, nil
}

// testTool is a simple Tool that echoes an input value.
type testTool struct{ called bool }

type testArgs struct {
	Text string `json:"text"`
}

func (t *testTool) Name() string        { return "echo" }
func (t *testTool) Description() string { return "echo tool" }
func (t *testTool) Parameters() any     { return &testArgs{} }
func (t *testTool) Execute(ctx context.Context, args any) (any, error) {
	t.called = true
	a := args.(*testArgs)
	return map[string]any{"echo": a.Text}, nil
}

func newTestRouter(models map[string]config.ModelConfig, fake RawClient) *router {
	r := &router{
		models:       models,
		clients:      make(map[string]RawClient),
		maxToolTurns: 3,
		logger:       slog.Default(),
	}
	// Pre-inject fake for the provider of the first model, and for both providers in models
	provs := map[string]struct{}{}
	for _, mc := range models {
		provs[mc.Provider] = struct{}{}
	}
	for p := range provs {
		r.clients[p] = fake
	}
	return r
}

// --- Sample tools for the requested workflow ---

type getUserLocationArgs struct{}

type getUserLocationTool struct{ called bool }

func (t *getUserLocationTool) Name() string { return "GetUserLocation" }
func (t *getUserLocationTool) Description() string {
	return "Returns the user's current city and state"
}
func (t *getUserLocationTool) Parameters() any { return &getUserLocationArgs{} }
func (t *getUserLocationTool) Execute(ctx context.Context, args any) (any, error) {
	t.called = true
	return map[string]any{"location": "Portland, Oregon"}, nil
}

type getWeatherArgs struct {
	Location string `json:"location"`
}

type getWeatherTool struct{ called bool }

func (t *getWeatherTool) Name() string        { return "GetWeatherInLocation" }
func (t *getWeatherTool) Description() string { return "Returns current weather for a location" }
func (t *getWeatherTool) Parameters() any     { return &getWeatherArgs{} }
func (t *getWeatherTool) Execute(ctx context.Context, args any) (any, error) {
	t.called = true
	a := args.(*getWeatherArgs)
	// Hardcoded weather string using provided location
	return map[string]any{"weather": "Sunny and mild in " + a.Location}, nil
}

func TestToolWorkflow_UserLocationThenWeather(t *testing.T) {
	// Fake provider issues two tool calls in sequence, then returns final JSON
	fc := &fakeClient{responses: []RawResponse{
		{ToolCalls: []core.ToolCall{{Name: "GetUserLocation", Args: []byte(`{}`)}}},
		{ToolCalls: []core.ToolCall{{Name: "GetWeatherInLocation", Args: []byte(`{"location":"Portland, Oregon"}`)}}},
		{Content: `{"weather":"Sunny and mild in Portland, Oregon"}`},
	}}
	models := map[string]config.ModelConfig{
		"g": {Provider: "gemini", Model: "gemini-1.5-pro", SupportsStructuredOutput: true, SupportsTools: true},
	}
	r := newTestRouter(models, fc)

	locTool := &getUserLocationTool{}
	weatherTool := &getWeatherTool{}

	// Ask the LLM to determine location, then get weather in that location
	raw, err := r.ExecuteRaw(context.Background(), Request{
		Model:    "g",
		Messages: []Message{{Role: RoleUser, Content: "Determine my location, then fetch the weather for it."}},
		Tools:    []Tool{locTool, weatherTool},
	})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if raw != `{"weather":"Sunny and mild in Portland, Oregon"}` {
		t.Fatalf("unexpected final JSON: %s", raw)
	}
	if !locTool.called {
		t.Fatalf("expected GetUserLocation to be called")
	}
	if !weatherTool.called {
		t.Fatalf("expected GetWeatherInLocation to be called")
	}
	if fc.calls != 3 {
		t.Fatalf("expected 3 model calls, got %d", fc.calls)
	}
}

func TestExecuteRaw_FinalString_NoTools(t *testing.T) {
	fc := &fakeClient{responses: []RawResponse{{Content: `{"ok":true}`}}}
	models := map[string]config.ModelConfig{
		"gpt4o": {Provider: "openai", Model: "gpt-4o", SupportsStructuredOutput: true, SupportsTools: true},
	}
	r := newTestRouter(models, fc)
	out, err := r.ExecuteRaw(context.Background(), Request{Model: "gpt4o", Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if out != `{"ok":true}` {
		t.Fatalf("unexpected out: %s", out)
	}
}

func TestExecute_Typed_Unmarshal(t *testing.T) {
	fc := &fakeClient{responses: []RawResponse{{Content: `{"x":1}`}}}
	models := map[string]config.ModelConfig{
		"g": {Provider: "gemini", Model: "gemini-1.5-pro", SupportsStructuredOutput: true, SupportsTools: true},
	}
	r := newTestRouter(models, fc)

	type R struct {
		X int `json:"x"`
	}
	got, err := Execute[R](context.Background(), r, Request{Model: "g", Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got.X != 1 {
		t.Fatalf("unexpected: %+v", got)
	}
}

func TestExecute_Typed_Repair(t *testing.T) {
	fc := &fakeClient{responses: []RawResponse{{Content: "```json\n{\"x\":2}\n```"}}}
	models := map[string]config.ModelConfig{
		"g": {Provider: "gemini", Model: "gemini-1.5-pro", SupportsStructuredOutput: true, SupportsTools: true},
	}
	r := newTestRouter(models, fc)
	type R struct {
		X int `json:"x"`
	}
	got, err := Execute[R](context.Background(), r, Request{Model: "g", Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got.X != 2 {
		t.Fatalf("unexpected: %+v", got)
	}
}

func TestToolLoop_Sequential(t *testing.T) {
	// First call requests tool; second call returns final JSON
	fc := &fakeClient{responses: []RawResponse{
		{ToolCalls: []core.ToolCall{{Name: "echo", Args: []byte(`{"text":"hello"}`)}}},
		{Content: `{"done":true}`},
	}}
	models := map[string]config.ModelConfig{
		"g": {Provider: "gemini", Model: "gemini-1.5-pro", SupportsStructuredOutput: true, SupportsTools: true},
	}
	r := newTestRouter(models, fc)
	tool := &testTool{}
	out, err := r.ExecuteRaw(context.Background(), Request{
		Model:    "g",
		Messages: []Message{{Role: RoleUser, Content: "use tool"}},
		Tools:    []Tool{tool},
	})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if out != `{"done":true}` {
		t.Fatalf("unexpected out: %s", out)
	}
	if !tool.called {
		t.Fatalf("tool was not called")
	}
	if fc.calls != 2 {
		t.Fatalf("expected 2 calls, got %d", fc.calls)
	}
}

func TestSelect_OpenAI_WebSuffix(t *testing.T) {
	fc := &fakeClient{responses: []RawResponse{{Content: `{"ok":true}`}}}
	models := map[string]config.ModelConfig{
		"gpt4o":     {Provider: "openai", Model: "gpt-4o", SupportsStructuredOutput: true, SupportsTools: true},
		"gpt4o-web": {Provider: "openai", Model: "gpt-4o-web", SupportsStructuredOutput: true, SupportsTools: true, SupportsWebSearch: true},
	}
	r := newTestRouter(models, fc)
	_, err := r.ExecuteRaw(context.Background(), Request{Model: "gpt4o", AllowWebSearch: true, Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if fc.lastModel != "gpt-4o-web" {
		t.Fatalf("expected web model, got %s", fc.lastModel)
	}
}

func TestSelect_OpenAI_WebVariantExplicit(t *testing.T) {
    fc := &fakeClient{responses: []RawResponse{{Content: `{"ok":true}`}}}
    models := map[string]config.ModelConfig{
        "gpt4o":     {Provider: "openai", Model: "gpt-4o", SupportsStructuredOutput: true, SupportsTools: true, WebVariant: "gpt4o-web"},
        "gpt4o-web": {Provider: "openai", Model: "gpt-4o-web", SupportsStructuredOutput: true, SupportsTools: true, SupportsWebSearch: true},
    }
    r := newTestRouter(models, fc)
    _, err := r.ExecuteRaw(context.Background(), Request{Model: "gpt4o", AllowWebSearch: true, Messages: []Message{{Role: RoleUser, Content: "hi"}}})
    if err != nil {
        t.Fatalf("unexpected err: %v", err)
    }
    if fc.lastModel != "gpt-4o-web" {
        t.Fatalf("expected explicit web variant model, got %s", fc.lastModel)
    }
}

func TestUnknownToolError(t *testing.T) {
	fc := &fakeClient{responses: []RawResponse{{ToolCalls: []core.ToolCall{{Name: "missing", Args: []byte(`{}`)}}}}}
	models := map[string]config.ModelConfig{
		"g": {Provider: "gemini", Model: "gemini-1.5-pro", SupportsStructuredOutput: true, SupportsTools: true},
	}
	r := newTestRouter(models, fc)
	_, err := r.ExecuteRaw(context.Background(), Request{Model: "g", Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if !errors.Is(err, moderr.ErrUnknownTool) {
		t.Fatalf("expected ErrUnknownTool, got %v", err)
	}
}

func TestBoundedInt(t *testing.T) {
	if boundedInt(0, 10) != 10 {
		t.Fatal("expected default to max")
	}
	if boundedInt(5, 10) != 5 {
		t.Fatal("expected 5")
	}
	if boundedInt(20, 10) != 10 {
		t.Fatal("expected cap to 10")
	}
}
