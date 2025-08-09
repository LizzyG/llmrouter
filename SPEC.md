## LLM Module Spec

### Purpose

A Go-native, multi-model LLM interface that provides:

- Typed final outputs via generics (`Execute[T]`)
- Centralized, provider-agnostic tool execution loop
- Automatic model/provider selection based on request + model capabilities
- Strongly typed tool parameter passing + JSON Schema generation
- Koanf-driven model registry (YAML/ConfigMap for prod, file/local for dev)
- Pure `net/http` provider adapters (no heavy SDKs)

Target Go version: 1.22

Module path: `github.com/lizzyg/llmrouter`

---

### Public API

#### Interfaces

```go
// Client is the only type applications use.
type Client interface {
    // Execute sends the request to the selected model, orchestrates tool calls if needed,
    // and returns a parsed result of type T.
    Execute[T any](ctx context.Context, req Request) (T, error)
}

// Tool is implemented by any callable function the model can invoke.
type Tool interface {
    Name() string
    Description() string
    // Parameters returns a POINTER to a zero-value struct for JSON schema generation + unmarshalling.
    Parameters() any
    Execute(ctx context.Context, args any) (any, error)
}
```

#### Request/Response

```go
type Request struct {
    Model          string
    Messages       []Message
    AllowWebSearch bool   // if true, router picks a search-capable model if needed
    Tools          []Tool // executable tool implementations
    MaxTokens      int
    Temperature    float32
    TopP           float32
    // Optional overrides
    Timeout        time.Duration // default applied if zero
}

type Message struct {
    Role    MessageRole
    Content string
    Images  []string // optional image URLs (assumed supported by all models in v1)
}

type MessageRole string
const (
    RoleSystem    MessageRole = "system"
    RoleUser      MessageRole = "user"
    RoleAssistant MessageRole = "assistant"
)
```

Structured output behavior:

- If `T` is not `string`, the final model content MUST be valid JSON parseable into `T`. If parsing fails, a lightweight JSON "repair" step is attempted, and on failure an error is returned.
- If `T` is `string`, the raw final content is returned (no JSON enforcement).
- Structured output is always required unless `T` is `string`.

---

### Config & Model Registry

#### YAML Format (example)

```yaml
llm:
  models:
    gpt4o:
      provider: openai
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}
      supports_web_search: false
      supports_tools: true
      supports_structured_output: true
      context_window: 128000
      max_output_tokens: 4000
    gpt4o-web:
      provider: openai
      model: gpt-4o-realtime-preview-web
      api_key: ${OPENAI_API_KEY}
      supports_web_search: true
      supports_tools: true
      supports_structured_output: true
      context_window: 128000
      max_output_tokens: 4000
    gemini15pro:
      provider: gemini
      model: gemini-1.5-pro
      api_key: ${GEMINI_API_KEY}
      supports_web_search: false
      supports_tools: true
      supports_structured_output: true
      context_window: 200000
      max_output_tokens: 4096
```

#### Go Structs

```go
type LLMConfig struct {
    Models map[string]ModelConfig `koanf:"models"`
}

type ModelConfig struct {
    Provider                 string `koanf:"provider"`
    Model                    string `koanf:"model"`
    APIKey                   string `koanf:"api_key"`
    SupportsWebSearch        bool   `koanf:"supports_web_search"`
    SupportsTools            bool   `koanf:"supports_tools"`
    SupportsStructuredOutput bool   `koanf:"supports_structured_output"`
    ContextWindow            int    `koanf:"context_window"`
    MaxOutputTokens          int    `koanf:"max_output_tokens"`
}
```

#### Loading

- Loaded once at startup (or first use) via Koanf.
- Default path: `config.yaml` at repository/application root; overridable via `LLM_CONFIG_PATH`.
- Environment expansion supported for secrets (e.g., `${OPENAI_API_KEY}`).

---

### Internal Components

#### Router (Model/Provider Selection)

```go
type router struct {
    models  map[string]ModelConfig
    clients map[string]RawClient // provider → singleton
    mu      sync.Mutex
    logger  *slog.Logger
    timeout time.Duration
}

func NewRouter(cfg LLMConfig, opts ...Option) Client
```

Selection logic:

1. If `req.Model != ""` → check it exists, use it. If `req.AllowWebSearch==true` and provider is `openai`, attempt to use a sibling model with `-web` suffix (e.g., `gpt4o-web`). Error if the `-web` model is not present. For non-OpenAI, require a model with `SupportsWebSearch=true` or error.
2. Else (no explicit model):
   - If `req.AllowWebSearch`, select the first model with `SupportsWebSearch=true` and any other required flags (e.g., tools).
   - If `len(req.Tools) > 0`, only choose models with `SupportsTools=true`.
3. Error if no matching model.

Get or Create `RawClient` (per provider singleton):

```go
func (r *router) getClient(mc ModelConfig) (RawClient, error) {
    r.mu.Lock()
    defer r.mu.Unlock()
    if c, ok := r.clients[mc.Provider]; ok {
        return c, nil
    }
    c, err := newProviderClient(mc)
    if err != nil { return nil, err }
    r.clients[mc.Provider] = c
    return c, nil
}
```

#### RawClient (Provider Adapter)

```go
type RawClient interface {
    // Single model call — no tool loop — may return tool call requests.
    Call(ctx context.Context, params CallParams) (RawResponse, error)
}

type CallParams struct {
    Model        string
    Messages     []Message
    ToolDefs     []ToolDef // names/descriptions/JSON schemas only
    OutputSchema string    // optional JSON schema for final output
    MaxTokens    int
    Temperature  float32
    TopP         float32
}

type ToolDef struct {
    Name        string
    Description string
    JSONSchema  string
}

type RawResponse struct {
    Content    string     // Final model text if no tool call
    ToolCalls  []ToolCall // Zero or more requested tool invocations
    Usage      Usage      // Token usage info
}

type ToolCall struct {
    Name string
    Args json.RawMessage
}

type Usage struct {
    PromptTokens     int
    CompletionTokens int
    TotalTokens      int
}
```

Providers implemented in v1: `openai`, `gemini` using pure `net/http`.

Retries with backoff on 429/5xx and a configurable default timeout.

#### Orchestrator (Central Tool Loop)

```go
func (r *router) Execute[T any](ctx context.Context, req Request) (T, error)
```

Behavior:

1. Select model using the rules above (including OpenAI `-web` mapping when `AllowWebSearch`).
2. Generate tool JSON Schemas from `Tool.Parameters()` using `invopop/jsonschema` (pointer-return enforced), and generate an output schema for `T` when `T != string`.
3. Enter a loop with `maxToolTurns` (default 3, configurable):
   - Call provider `Call` with messages, tool defs, and optional output schema.
   - If `resp.ToolCalls` is empty → final answer path:
     - If `T == string`, return raw text.
     - Else unmarshal JSON into `T`. If it fails, attempt a lightweight JSON repair then retry once.
   - Else sequentially execute each tool:
     - Find tool by name or return `ErrUnknownTool`.
     - Unmarshal tool args into the pointer obtained from `tool.Parameters()`.
     - Execute the tool and append the result back into the conversation as an assistant message using `formatToolResult`.
4. If turns exhausted, return `ErrMaxToolTurns`.

All requests and responses are logged via `slog` with `provider`, `model`, and `usage` included (never log API keys). Usage is always logged after each request.

---

### Special Handling: Web Search

- `WebSearchTool` (if implemented by the application) is treated like any other tool.
- Router uses `SupportsWebSearch` to pick a native search-capable model.
- OpenAI-specific: when `AllowWebSearch=true` and the selected provider is `openai`, attempt to use the same model name with a `-web` suffix (e.g., `gpt4o` → `gpt4o-web`). Error if such a model does not exist.

---

### Utilities

- `generateJSONSchema(obj any) string`: reflect on struct → JSON Schema (`invopop/jsonschema`).
- `findTool(tools []Tool, name string) Tool`: linear search/mapping.
- `boundedInt(req, max int) int`: enforce model’s hard max.
- `repairJSON(s string) (string, bool)`: attempt minimal fixups (strip fences, trim to outermost JSON object/array). Returns repaired string and a flag indicating modification.

---

### Error & Safety Considerations

- Limit `maxToolTurns` (default 3; configurable) to prevent infinite loops.
- Validate tool call names.
- Custom errors: `ErrNoMatchingModel`, `ErrUnknownTool`, `ErrMaxToolTurns`, `ErrStructuredOutput`.
- Never include API keys or sensitive data in logs or errors.

---

### Logging & Metrics

- Use `log/slog` for structured logs.
- Every request log includes `provider`, `model`, and post-call `usage`.
- Keep it simple for v1; no metrics library by default.

---

### Testing & CI

- Prefer unit tests for router selection, orchestrator loop, schema generation, and adapter marshaling.
- Integration tests for live providers are included and skipped by default; enable via build tag `integration` or by `go test -tags=integration`.
- No API keys appear in logs or errors during tests.

---

### Packaging & Layout

- Public API in root package `llmrouter`.
- Internal code under `internal/...` (providers, config, util).
- Examples under `examples/`.
- License: Non-commercial use (see `LICENSE`).

Suggested layout:

```
llmrouter/
  SPEC.md
  LICENSE
  README.md
  config.yaml.example
  go.mod
  internal/
    config/
      loader.go
    providers/
      openai/
        client.go
      gemini/
        client.go
    util/
      jsonschema.go
      jsonrepair.go
  errors/
    errors.go
  api.go          // public types & Client interface
  router.go       // NewRouter + Execute[T]
  examples/
    basic/
      main.go
```

---

### OpenAI Adapter (v1 outline)

- Endpoint: Responses API (or Chat Completions if necessary) via `net/http`.
- Map messages with optional image URLs.
- Support tools via tool definitions; surface tool calls back to orchestrator.
- When `AllowWebSearch` and selected model is OpenAI, use the `-web` model variant as per config.
- Include usage parsing when available.
- Retries with exponential backoff on 429/5xx; obey context deadline/timeout.

### Gemini Adapter (v1 outline)

- Endpoint: Generative Language API via `net/http`.
- Map messages and image URLs; support tools if available.
- Include usage parsing when available.
- Retries with exponential backoff on 429/5xx; obey context deadline/timeout.

---

### Example Usage (no tools)

```go
type Answer struct {
    Summary string `json:"summary"`
}

client := llmrouter.NewFromFile("config.yaml")

res, err := client.Execute[Answer](ctx, llmrouter.Request{
    Model: "gpt4o",
    Messages: []llmrouter.Message{{
        Role:    llmrouter.RoleUser,
        Content: "Summarize the latest news about Go 1.22",
    }},
    MaxTokens:   800,
    Temperature: 0.2,
})
```

---

### Defaults

- `maxToolTurns = 3`
- Request timeout: configurable default (applied if `Request.Timeout == 0`).
- `Temperature`, `TopP`, and `MaxTokens` default to provider/model defaults if zero.


