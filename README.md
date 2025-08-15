## llmrouter

Go-native, provider-agnostic LLM client with typed outputs and a centralized tool loop. Providers: OpenAI and Gemini (net/http).

### Install

```bash
go get github.com/lizzyg/llmrouter
```

### Config

Create `config.yaml` in your app root (or set `LLM_CONFIG_PATH`).

```yaml
llm:
  models:
    gpt4o:
      provider: openai
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}
      web_variant: gpt4o-web
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

**Note:** The `web_variant` field allows you to explicitly map a model to its web-enabled counterpart for when `AllowWebSearch=true` is requested. This is the recommended approach over relying on automatic `-web` suffix detection.

### Usage

```go
package main

import (
    "context"
    "fmt"
    llm "github.com/lizzyg/llmrouter"
)

type Answer struct { Summary string `json:"summary"` }

func main() {
    client, err := llm.NewFromFile()
    if err != nil { panic(err) }
    ctx := context.Background()

    res, err := client.Execute[Answer](ctx, llm.Request{
        Model: "gpt4o",
        Messages: []llm.Message{{
            Role:    llm.RoleUser,
            Content: "Summarize Go 1.22 features as JSON",
        }},
        MaxTokens:   500,
        Temperature: 0.2,
    })
    if err != nil { panic(err) }
    fmt.Println(res.Summary)
}
```

### Notes

- If `AllowWebSearch=true` for an OpenAI model, the router will look for a web-enabled variant. You can explicitly specify this using the `web_variant` field in `config.yaml` (recommended), or it will fallback to looking for a `-web` suffix variant (e.g., `gpt4o-web`).
- If `T` is not `string`, final content must be JSON (repair attempted on failure).
- Tool calls are executed sequentially.

### License

Non-commercial use only. See `LICENSE`.


