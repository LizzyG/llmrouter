package llmrouter

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"time"

	moderr "github.com/lizzyg/llmrouter/errors"
	"github.com/lizzyg/llmrouter/internal/config"
	"github.com/lizzyg/llmrouter/internal/core"
	provfactory "github.com/lizzyg/llmrouter/internal/providers"
	"github.com/lizzyg/llmrouter/internal/util"
)

// RawClient is implemented by provider adapters.
type RawClient = core.RawClient
type CallParams = core.CallParams
type ToolDef = core.ToolDef
type RawResponse = core.RawResponse
type Usage = core.Usage
type ToolCall = core.ToolCall

type router struct {
	models       map[string]config.ModelConfig
	clients      map[string]RawClient // provider -> singleton client
	logger       *slog.Logger
	httpClient   *http.Client
	maxToolTurns int
}

// Option allows functional configuration.
type Option func(*router)

// WithLogger sets a custom slog logger.
func WithLogger(l *slog.Logger) Option { return func(r *router) { r.logger = l } }

// WithHTTPClient sets a custom http.Client.
func WithHTTPClient(c *http.Client) Option { return func(r *router) { r.httpClient = c } }

// WithMaxToolTurns sets the maximum tool turns.
func WithMaxToolTurns(n int) Option { return func(r *router) { r.maxToolTurns = n } }

// NewFromFile loads config via internal/config.Load and returns a Client.
func NewFromFile() (Client, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}
	return NewRouter(*cfg), nil
}

// NewRouter builds a router from config and options.
func NewRouter(cfg config.LLMConfig, opts ...Option) Client {
	r := &router{
		models:       cfg.Models,
		clients:      make(map[string]RawClient),
		logger:       slog.Default(),
		httpClient:   &http.Client{Timeout: 30 * time.Second},
		maxToolTurns: 5,
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

// ExecuteRaw is the orchestrator (tool loop) and returns the final JSON string.
func (r *router) ExecuteRaw(ctx context.Context, req Request) (string, error) {
	mc, modelKey, err := r.selectModel(req)
	if err != nil {
		return "", err
	}

	rc, err := r.getClient(mc)
	if err != nil {
		return "", err
	}

	// Prepare tool definitions for the API
	defs := make([]ToolDef, len(req.Tools))
	for i, t := range req.Tools {
		// Reflect tool parameters into a JSON schema fragment per field using invopop
		// by leveraging util.GenerateToolJSONSchema, then convert into a parameter list.
		schemaStr := util.GenerateToolJSONSchema(t.Parameters())
		// Parse schema into parameter list: properties + required
		var schema map[string]any
		_ = json.Unmarshal([]byte(schemaStr), &schema)
		props, _ := schema["properties"].(map[string]any)
		reqList := map[string]bool{}
		if reqArr, ok := schema["required"].([]any); ok {
			for _, v := range reqArr {
				if s, ok2 := v.(string); ok2 {
					reqList[s] = true
				}
			}
		}
		paramList := make([]core.ToolParameter, 0, len(props))
		for name, frag := range props {
			var fragMap map[string]any
			if m, ok := frag.(map[string]any); ok {
				fragMap = m
			} else {
				fragMap = map[string]any{"type": "string"}
			}
			paramList = append(paramList, core.ToolParameter{
				Name:     name,
				Required: reqList[name],
				Schema:   fragMap,
			})
		}
		defs[i] = ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  paramList,
		}
	}

	// carry any explicit schema provided by typed helper for providers that support it
	var outputSchema string

	conversation := req.Messages
	maxTurns := r.maxToolTurns
	if maxTurns <= 0 {
		maxTurns = 3
	}
	for turn := 0; turn < maxTurns; turn++ {
		// Respect per-request timeout if provided
		callCtx := ctx
		var cancel context.CancelFunc
		if req.Timeout > 0 {
			callCtx, cancel = context.WithTimeout(ctx, req.Timeout)
			defer cancel()
		}
		if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
			r.logger.Info("outgoing messages",
				slog.String("provider", mc.Provider),
				slog.String("model", mc.Model),
				slog.Any("messages", conversation),
				slog.Any("tools", defs),
			)
		}
		start := time.Now()
		resp, callErr := rc.Call(callCtx, CallParams{
			Model:        mc.Model,
			Messages:     mapMessages(conversation),
			ToolDefs:     defs,
			OutputSchema: outputSchema,
			MaxTokens:    boundedInt(req.MaxTokens, mc.MaxOutputTokens),
			Temperature:  req.Temperature,
			TopP:         req.TopP,
		})
		duration := time.Since(start)

		r.logger.Info("llm call",
			slog.String("provider", mc.Provider),
			slog.String("model", mc.Model),
			slog.String("model_key", modelKey),
			slog.Int("prompt_tokens", resp.Usage.PromptTokens),
			slog.Int("completion_tokens", resp.Usage.CompletionTokens),
			slog.Int("total_tokens", resp.Usage.TotalTokens),
			slog.Duration("latency_ms", duration),
			slog.Bool("error", callErr != nil),
		)

		if callErr != nil {
			return "", callErr
		}

		// STOP: No tool call → Final answer
		if len(resp.ToolCalls) == 0 {
			return resp.Content, nil
		}

		// Surface the model's function calls back into the conversation so
		// provider adapters like Gemini can pair them with subsequent tool responses.
		if len(resp.ToolCalls) > 0 {
			fcList := make([]map[string]any, 0, len(resp.ToolCalls))
			for _, tc := range resp.ToolCalls {
				var args any
				_ = json.Unmarshal(tc.Args, &args)
				fcList = append(fcList, map[string]any{
					"tool": tc.Name,
					"args": args,
				})
			}
			if b, err := json.Marshal(fcList); err == nil {
				conversation = append(conversation, Message{Role: RoleAssistant, Content: string(b)})
			}
		}

		// EXECUTE TOOLS sequentially and collect all results
		var toolResults []map[string]any
		for _, tc := range resp.ToolCalls {
			tool := findTool(req.Tools, tc.Name)
			if tool == nil {
				return "", moderr.ErrUnknownTool
			}
			argStruct := tool.Parameters()
			if err := json.Unmarshal(tc.Args, argStruct); err != nil {
				return "", err
			}
			output, err := tool.Execute(ctx, argStruct)
			if err != nil {
				return "", err
			}
			if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
				r.logger.Info("tool executed",
					slog.String("tool", tc.Name),
					slog.Any("args", argStruct),
					slog.Any("output", output),
				)
			}
			// Store tool results in a format that Gemini can parse as functionResponse
			toolResults = append(toolResults, map[string]any{
				"tool":   tc.Name,
				"result": output,
			})
		}

		// Add all tool results as a single assistant message
		if len(toolResults) > 0 {
			// Format multiple tool results as a JSON array for consistent parsing
			b, _ := json.Marshal(toolResults)
			if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
				r.logger.Info("combined tool results",
					slog.Int("count", len(toolResults)),
					slog.String("content", string(b)),
				)
			}
			conversation = append(conversation, Message{
				Role:    RoleAssistant,
				Content: string(b),
			})
		}
	}
	return "", moderr.ErrMaxToolTurns
}

func (r *router) getClient(mc config.ModelConfig) (RawClient, error) {
	key := mc.Provider
	if c, ok := r.clients[key]; ok {
		return c, nil
	}
	c, err := provfactory.NewProviderClient(mc, r.httpClient, r.logger)
	if err != nil {
		return nil, err
	}
	r.clients[key] = c
	return c, nil
}

func (r *router) selectModel(req Request) (config.ModelConfig, string, error) {
	// If user specified a model, use it.
	if req.Model != "" {
		mc, ok := r.models[req.Model]
		if !ok {
			return config.ModelConfig{}, "", moderr.ErrNoMatchingModel
		}
		if req.AllowWebSearch && mc.Provider == "openai" {
			// openai: switch to -web variant of the same model key (e.g., gpt4o -> gpt4o-web)
			webKey := req.Model + "-web"
			webModel, ok := r.models[webKey]
			if !ok {
				return config.ModelConfig{}, "", moderr.ErrNoMatchingModel
			}
			return webModel, webKey, nil
		}
		// Validate tool requirement and web search availability when explicitly chosen
		if len(req.Tools) > 0 && !mc.SupportsTools {
			return config.ModelConfig{}, "", moderr.ErrNoMatchingModel
		}
		if req.AllowWebSearch && !mc.SupportsWebSearch {
			return config.ModelConfig{}, "", moderr.ErrNoMatchingModel
		}
		return mc, req.Model, nil
	}

	// Auto select
	for key, mc := range r.models {
		if req.AllowWebSearch && !mc.SupportsWebSearch {
			continue
		}
		if len(req.Tools) > 0 && !mc.SupportsTools {
			continue
		}
		return mc, key, nil
	}
	return config.ModelConfig{}, "", moderr.ErrNoMatchingModel
}

func formatToolResult(name string, output any) string {
	b, _ := json.Marshal(map[string]any{
		"tool":   name,
		"result": output,
	})
	return string(b)
}

func findTool(tools []Tool, name string) Tool {
	for _, t := range tools {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

func boundedInt(req, max int) int {
	if max <= 0 {
		return req
	}
	if req <= 0 {
		return max
	}
	if req > max {
		return max
	}
	return req
}

func mapMessages(msgs []Message) []core.Message {
	out := make([]core.Message, len(msgs))
	for i, m := range msgs {
		out[i] = core.Message{
			Role:    string(m.Role),
			Content: m.Content,
			Images:  append([]string(nil), m.Images...),
		}
	}
	return out
}

// executeWithSchema allows the typed helper to pass an explicit output schema.
// It returns the raw content string from the provider.
func (r *router) executeWithSchema(ctx context.Context, req Request, outputSchema string, requireStructured bool) (string, error) {
	mc, modelKey, err := r.selectModel(req)
	if err != nil {
		return "", err
	}
	rc, err := r.getClient(mc)
	if err != nil {
		return "", err
	}

	defs := make([]ToolDef, len(req.Tools))
	for i, t := range req.Tools {
		// Build parameter list from reflected schema
		schemaStr := util.GenerateToolJSONSchema(t.Parameters())
		var schema map[string]any
		_ = json.Unmarshal([]byte(schemaStr), &schema)
		props, _ := schema["properties"].(map[string]any)
		reqList := map[string]bool{}
		if reqArr, ok := schema["required"].([]any); ok {
			for _, v := range reqArr {
				if s, ok2 := v.(string); ok2 {
					reqList[s] = true
				}
			}
		}
		paramList := make([]core.ToolParameter, 0, len(props))
		for name, frag := range props {
			var fragMap map[string]any
			if m, ok := frag.(map[string]any); ok {
				fragMap = m
			} else {
				fragMap = map[string]any{"type": "string"}
			}
			paramList = append(paramList, core.ToolParameter{
				Name:     name,
				Required: reqList[name],
				Schema:   fragMap,
			})
		}
		defs[i] = ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  paramList,
		}
	}

	// Only pass schema through if provider supports it; otherwise leave empty and we will parse/repair after.
	if !mc.SupportsStructuredOutput {
		outputSchema = ""
	} else if outputSchema != "" {
		// Inline $ref and strip draft keys to ensure providers like Gemini accept it
		outputSchema = util.SanitizeResponseSchemaJSON(outputSchema)
	}

	conversation := req.Messages
	maxTurns := r.maxToolTurns
	if maxTurns <= 0 {
		maxTurns = 3
	}
	for turn := 0; turn < maxTurns; turn++ {
		callCtx := ctx
		var cancel context.CancelFunc
		if req.Timeout > 0 {
			callCtx, cancel = context.WithTimeout(ctx, req.Timeout)
			defer cancel()
		}
		start := time.Now()
		resp, callErr := rc.Call(callCtx, CallParams{
			Model:        mc.Model,
			Messages:     mapMessages(conversation),
			ToolDefs:     defs,
			OutputSchema: outputSchema,
			MaxTokens:    boundedInt(req.MaxTokens, mc.MaxOutputTokens),
			Temperature:  req.Temperature,
			TopP:         req.TopP,
		})
		duration := time.Since(start)

		r.logger.Info("llm call",
			slog.String("provider", mc.Provider),
			slog.String("model", mc.Model),
			slog.String("model_key", modelKey),
			slog.Int("prompt_tokens", resp.Usage.PromptTokens),
			slog.Int("completion_tokens", resp.Usage.CompletionTokens),
			slog.Int("total_tokens", resp.Usage.TotalTokens),
			slog.Duration("latency_ms", duration),
			slog.Bool("error", callErr != nil),
		)

		if callErr != nil {
			return "", callErr
		}

		if len(resp.ToolCalls) == 0 {
			return resp.Content, nil
		}

		// Surface the model's function calls back into the conversation so
		// provider adapters like Gemini can pair them with subsequent tool responses.
		if len(resp.ToolCalls) > 0 {
			fcList := make([]map[string]any, 0, len(resp.ToolCalls))
			for _, tc := range resp.ToolCalls {
				var args any
				_ = json.Unmarshal(tc.Args, &args)
				fcList = append(fcList, map[string]any{
					"tool": tc.Name,
					"args": args,
				})
			}
			if b, err := json.Marshal(fcList); err == nil {
				conversation = append(conversation, Message{Role: RoleAssistant, Content: string(b)})
			}
		}

		// EXECUTE TOOLS sequentially and collect all results
		var toolResults []map[string]any
		for _, tc := range resp.ToolCalls {
			tool := findTool(req.Tools, tc.Name)
			if tool == nil {
				return "", moderr.ErrUnknownTool
			}
			argStruct := tool.Parameters()
			if err := json.Unmarshal(tc.Args, argStruct); err != nil {
				return "", err
			}
			output, err := tool.Execute(callCtx, argStruct)
			if err != nil {
				return "", err
			}
			if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
				r.logger.Info("tool executed",
					slog.String("tool", tc.Name),
					slog.Any("args", argStruct),
					slog.Any("output", output),
				)
			}
			// Store tool results in a format that Gemini can parse as functionResponse
			toolResults = append(toolResults, map[string]any{
				"tool":   tc.Name,
				"result": output,
			})
		}

		// Add all tool results as a single assistant message
		if len(toolResults) > 0 {
			// Format multiple tool results as a JSON array for consistent parsing
			b, _ := json.Marshal(toolResults)
			if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
				r.logger.Info("combined tool results",
					slog.Int("count", len(toolResults)),
					slog.String("content", string(b)),
				)
			}
			conversation = append(conversation, Message{
				Role:    RoleAssistant,
				Content: string(b),
			})
		}
	}
	return "", moderr.ErrMaxToolTurns
}
