package llmrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"sort"
	"sync"
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

type router struct {
	models       map[string]config.ModelConfig
	clients      map[string]RawClient // provider -> singleton client
	logger       *slog.Logger
	httpClient   *http.Client
	maxToolTurns int
	mu           sync.Mutex
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

// executeInternal consolidates ExecuteRaw and executeWithSchema to avoid duplication.
// When requireStructured is true and the provider supports it, outputSchema is sanitized and passed through.
func (r *router) executeInternal(ctx context.Context, req Request, outputSchema string, requireStructured bool) (string, error) {
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
		// Generate tool parameters directly from the struct using reflection
		paramMaps, err := util.GenerateToolParameters(t.Parameters())
		if err != nil {
			return "", err
		}
		
		// Convert the parameter maps to core.ToolParameter structs
		paramList := make([]core.ToolParameter, 0, len(paramMaps))
		for _, paramMap := range paramMaps {
			name := paramMap["name"].(string)
			required := paramMap["required"].(bool)
			description := paramMap["description"].(string)
			schema := paramMap["schema"].(map[string]any)
			
			paramList = append(paramList, core.ToolParameter{
				Name:        name,
				Required:    required,
				Description: description,
				Schema:      schema,
			})
		}
		
		defs[i] = ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  paramList,
		}
	}

	// Only pass schema through if required and provider supports it; otherwise leave empty and we will parse/repair after.
	if !requireStructured || !mc.SupportsStructuredOutput {
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
		result, done, err := func() (string, bool, error) {
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
			messages, err := r.mapMessages(conversation)
			if err != nil {
				return "", false, err
			}
			resp, callErr := rc.Call(callCtx, CallParams{
				Model:        mc.Model,
				Messages:     messages,
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
				return "", true, callErr
			}

			// STOP: No tool call → Final answer
			if len(resp.ToolCalls) == 0 {
				return resp.Content, true, nil
			}

			// Surface the model's function calls back into the conversation so
			// provider adapters like Gemini can pair them with subsequent tool responses.
			if len(resp.ToolCalls) > 0 {
				// Record structured tool calls directly in the conversation
				toolCalls := make([]ToolCall, len(resp.ToolCalls))
				for i, tc := range resp.ToolCalls {
					var args any
					if len(tc.Args) > 0 {
						if err := json.Unmarshal(tc.Args, &args); err != nil {
							r.logger.Warn("failed to unmarshal tool call args from provider response", "error", err, "tool", tc.Name)
						}
					}
					toolCalls[i] = ToolCall{CallID: tc.CallID, Name: tc.Name, Args: args}
				}
				conversation = append(conversation, Message{Role: RoleAssistant, ToolCalls: toolCalls})
			}

			// EXECUTE TOOLS sequentially and collect all results
			var toolResults []map[string]any
			for _, tc := range resp.ToolCalls {
				tool := findTool(req.Tools, tc.Name)
				if tool == nil {
					return "", true, moderr.ErrUnknownTool
				}
				argStruct := tool.Parameters()
				if err := json.Unmarshal(tc.Args, argStruct); err != nil {
					return "", true, err
				}
				output, err := tool.Execute(callCtx, argStruct)
				if err != nil {
					return "", true, err
				}
				if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
					r.logger.Info("tool executed",
						slog.String("tool", tc.Name),
						slog.Any("args", argStruct),
						slog.Any("output", output),
					)
				}
				// Store tool results in a format that Gemini (functionResponse) and OpenAI (tool message) can parse
				item := map[string]any{
					"tool":   tc.Name,
					"result": output,
				}
				if tc.CallID != "" {
					item["tool_call_id"] = tc.CallID
				}
				toolResults = append(toolResults, item)
			}

			// Add all tool results as a single assistant message using structured field
			if len(toolResults) > 0 {
				if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
					r.logger.Info("combined tool results",
						slog.Int("count", len(toolResults)),
					)
				}
				// Convert to []ToolResult for the public message type
				tr := make([]ToolResult, 0, len(toolResults))
				for _, it := range toolResults {
					tr = append(tr, ToolResult{
						CallID: asString(it["tool_call_id"]),
						Name:   asString(it["tool"]),
						Result: it["result"],
					})
				}
				conversation = append(conversation, Message{Role: RoleAssistant, ToolResults: tr})
			}
			return "", false, nil
		}()
		if done {
			return result, err
		}
	}
	return "", moderr.ErrMaxToolTurns
}

// ExecuteRaw is the orchestrator (tool loop) and returns the final JSON string.
func (r *router) ExecuteRaw(ctx context.Context, req Request) (string, error) {
	return r.executeInternal(ctx, req, "", false)
}

func (r *router) getClient(mc config.ModelConfig) (RawClient, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

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
			// Prefer explicit web variant mapping from configuration over suffix conventions
			if mc.WebVariant != "" {
				if webModel, ok := r.models[mc.WebVariant]; ok {
					return webModel, mc.WebVariant, nil
				}
				return config.ModelConfig{}, "", moderr.ErrNoMatchingModel
			}
			// Fallback: maintain backward-compat with "-web" suffix if present in config
			fallbackKey := req.Model + "-web"
			if webModel, ok := r.models[fallbackKey]; ok {
				return webModel, fallbackKey, nil
			}
			return config.ModelConfig{}, "", moderr.ErrNoMatchingModel
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

	// Auto select - use sorted keys for deterministic behavior
	keys := make([]string, 0, len(r.models))
	for k := range r.models {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		mc := r.models[key]
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

// formatToolResult was unused; removed to avoid dead code

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

func (r *router) mapMessages(msgs []Message) ([]core.Message, error) {
	out := make([]core.Message, len(msgs))
	for i, m := range msgs {
		toolCalls, err := mapToolCalls(m.ToolCalls, r.logger)
		if err != nil {
			return nil, err
		}
		out[i] = core.Message{
			Role:        string(m.Role),
			Content:     m.Content,
			Images:      append([]string(nil), m.Images...),
			ToolCalls:   toolCalls,
			ToolResults: mapToolResults(m.ToolResults),
		}
	}
	return out, nil
}

func mapToolCalls(in []ToolCall, logger *slog.Logger) ([]core.ToolCall, error) {
	if len(in) == 0 {
		return nil, nil
	}
	out := make([]core.ToolCall, 0, len(in))
	for _, tc := range in {
		var raw json.RawMessage
		if tc.Args != nil {
			b, err := json.Marshal(tc.Args)
			if err != nil {
				logger.Error("failed to marshal tool call args for core message", "error", err, "tool", tc.Name)
				return nil, fmt.Errorf("failed to marshal tool call args for tool %s: %w", tc.Name, err)
			}
			raw = b
		}
		out = append(out, core.ToolCall{CallID: tc.CallID, Name: tc.Name, Args: raw})
	}
	return out, nil
}

func mapToolResults(in []ToolResult) []core.ToolResult {
	if len(in) == 0 {
		return nil
	}
	out := make([]core.ToolResult, len(in))
	for i, tr := range in {
		out[i] = core.ToolResult{CallID: tr.CallID, Name: tr.Name, Result: tr.Result}
	}
	return out
}

func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// executeWithSchema allows the typed helper to pass an explicit output schema.
// It returns the raw content string from the provider.
func (r *router) executeWithSchema(ctx context.Context, req Request, outputSchema string, requireStructured bool) (string, error) {
	return r.executeInternal(ctx, req, outputSchema, requireStructured)
}
