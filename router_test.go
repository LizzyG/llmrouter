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

// mockTestTool implements the Tool interface for testing
type mockTestTool struct{}

func (m *mockTestTool) Name() string        { return "TestTool" }
func (m *mockTestTool) Description() string { return "A test tool" }
func (m *mockTestTool) Parameters() any     { return struct{}{} }
func (m *mockTestTool) Execute(ctx context.Context, args any) (any, error) {
	return "test result", nil
}

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

func TestMapToolCalls_HandlesMarshalError(t *testing.T) {
	logger := slog.Default()
	
	// Create a tool call with unmarshalable args (channels can't be marshaled)
	ch := make(chan int)
	toolCalls := []ToolCall{{
		CallID: "test1",
		Name:   "ValidTool",
		Args:   map[string]any{"data": "valid"},
	}, {
		CallID: "test2", 
		Name:   "InvalidTool",
		Args:   ch, // This will fail to marshal
	}}
	
	result, err := mapToolCalls(toolCalls, logger)
	
	// Should return an error for unmarshalable args
	if err == nil {
		t.Fatal("expected error for unmarshalable tool call args")
	}
	if result != nil {
		t.Fatalf("expected nil result when error occurs, got %v", result)
	}
}

func TestMapToolCalls_ValidArgs(t *testing.T) {
	logger := slog.Default()
	
	// Create tool calls with valid args
	toolCalls := []ToolCall{{
		CallID: "test1",
		Name:   "ValidTool1",
		Args:   map[string]any{"data": "valid"},
	}, {
		CallID: "test2", 
		Name:   "ValidTool2",
		Args:   nil, // nil args should be fine
	}, {
		CallID: "test3",
		Name:   "ValidTool3", 
		Args:   []string{"arg1", "arg2"},
	}}
	
	result, err := mapToolCalls(toolCalls, logger)
	
	// Should succeed for valid args
	if err != nil {
		t.Fatalf("unexpected error for valid tool call args: %v", err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 tool calls, got %d", len(result))
	}
	if result[0].Name != "ValidTool1" {
		t.Fatalf("expected ValidTool1, got %s", result[0].Name)
	}
}

func TestSelectModel_DeterministicAutoSelection(t *testing.T) {
	// Test that auto-selection is deterministic by using multiple models
	// and ensuring the same one is always selected
	models := map[string]config.ModelConfig{
		"zebra": {
			Provider:     "openai",
			Model:        "gpt-4",
			SupportsTools: true,
		},
		"alpha": {
			Provider:     "openai", 
			Model:        "gpt-3.5",
			SupportsTools: true,
		},
		"beta": {
			Provider:     "gemini",
			Model:        "gemini-pro",
			SupportsTools: true,
		},
	}
	
	r := &router{models: models}
	
	// Create a mock tool that implements the Tool interface
	mockTool := &mockTestTool{}
	
	// Run selection multiple times and ensure same result
	var firstKey string
	for i := 0; i < 10; i++ {
		_, key, err := r.selectModel(Request{Tools: []Tool{mockTool}})
		if err != nil {
			t.Fatalf("selectModel failed: %v", err)
		}
		if i == 0 {
			firstKey = key
		} else if key != firstKey {
			t.Fatalf("selectModel is non-deterministic: got %s on iteration %d, expected %s", key, i, firstKey)
		}
	}
	
	// Should always select "alpha" (first alphabetically that supports tools)
	if firstKey != "alpha" {
		t.Fatalf("expected alpha to be selected (alphabetically first), got %s", firstKey)
	}
}
