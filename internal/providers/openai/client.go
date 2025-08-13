package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"time"

	"github.com/lizzyg/llmrouter/internal/config"
	"github.com/lizzyg/llmrouter/internal/core"
)

type Client struct {
	apiKey     string
	httpClient *http.Client
	logger     *slog.Logger
	model      string
}

func New(mc config.ModelConfig, hc *http.Client, logger *slog.Logger) *Client {
	return &Client{
		apiKey:     mc.APIKey,
		httpClient: hc,
		logger:     logger,
		model:      mc.Model,
	}
}

type chatRequest struct {
	Model          string           `json:"model"`
	Messages       []map[string]any `json:"messages"`
	Tools          []map[string]any `json:"tools,omitempty"`
	MaxTokens      int              `json:"max_tokens,omitempty"`
	Temperature    float32          `json:"temperature,omitempty"`
	TopP           float32          `json:"top_p,omitempty"`
	ResponseFormat map[string]any   `json:"response_format,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   any `json:"content"`
			ToolCalls []struct {
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type toolCallItem struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"arguments"`
}

func (c *Client) Call(ctx context.Context, params core.CallParams) (core.RawResponse, error) {
	payload := chatRequest{
		Model:       params.Model,
		Messages:    mapChatMessages(params.Messages),
		MaxTokens:   params.MaxTokens,
		Temperature: params.Temperature,
		TopP:        params.TopP,
	}
	if len(params.ToolDefs) > 0 {
		payload.Tools = mapTools(params.ToolDefs)
	}
	if params.OutputSchema != "" {
		// Chat Completions supports json_object enforcement (not full schema). Use it when schema requested.
		payload.ResponseFormat = map[string]any{"type": "json_object"}
	}

	body, _ := json.Marshal(payload)

	var rr chatResponse
	err := c.withRetry(ctx, func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			b, _ := io.ReadAll(resp.Body)
			return fmt.Errorf("openai http %d: %s", resp.StatusCode, string(b))
		}
		dec := json.NewDecoder(resp.Body)
		return dec.Decode(&rr)
	})
	if err != nil {
		return core.RawResponse{}, err
	}

	out := core.RawResponse{}
	if len(rr.Choices) > 0 {
		msg := rr.Choices[0].Message
		if len(msg.ToolCalls) > 0 {
			out.ToolCalls = make([]core.ToolCall, len(msg.ToolCalls))
			for i, tc := range msg.ToolCalls {
				out.ToolCalls[i] = core.ToolCall{Name: tc.Function.Name, Args: json.RawMessage(tc.Function.Arguments)}
			}
		} else {
			switch v := msg.Content.(type) {
			case string:
				out.Content = v
			case []any:
				// concatenate text parts
				var acc string
				for _, p := range v {
					if m, ok := p.(map[string]any); ok {
						if m["type"] == "text" {
							if s, ok2 := m["text"].(string); ok2 {
								if acc == "" {
									acc = s
								} else {
									acc += "\n" + s
								}
							}
						}
					}
				}
				out.Content = acc
			default:
				out.Content = ""
			}
		}
	}
	out.Usage = core.Usage{PromptTokens: rr.Usage.PromptTokens, CompletionTokens: rr.Usage.CompletionTokens, TotalTokens: rr.Usage.TotalTokens}
	return out, nil
}

func mapChatMessages(msgs []core.Message) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		content := []any{}
		if m.Content != "" {
			content = append(content, map[string]any{"type": "text", "text": m.Content})
		}
		for _, img := range m.Images {
			content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": img}})
		}
		out = append(out, map[string]any{
			"role":    m.Role,
			"content": content,
		})
	}
	return out
}

func mapTools(defs []core.ToolDef) []map[string]any {
	out := make([]map[string]any, len(defs))
	for i, d := range defs {
		// Build JSON schema from parameter list to avoid provider-specific leakage
		schema := core.GenerateJSONSchemaFromToolDef(d)
		params := coerceOpenAIParams(schema)
		out[i] = map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        d.Name,
				"description": d.Description,
				"parameters":  params,
			},
		}
	}
	return out
}

// coerceOpenAIParams ensures the parameters JSON meets Chat Completions expectations
// for a function JSON Schema (must be type: object at top-level).
func coerceOpenAIParams(schema string) any {
	var m map[string]any
	if err := json.Unmarshal([]byte(schema), &m); err != nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	if t, ok := m["type"].(string); !ok || t == "" || t == "null" {
		m["type"] = "object"
	}
	if m["type"] != "object" {
		m["type"] = "object"
	}
	if _, ok := m["properties"]; !ok {
		m["properties"] = map[string]any{}
	}
	return m
}

// withRetry performs exponential backoff retries on transient errors.
func (c *Client) withRetry(ctx context.Context, fn func() error) error {
	const (
		maxAttempts = 5
		baseDelay   = 200 * time.Millisecond
		maxDelay    = 3 * time.Second
	)
	var attempt int
	for {
		err := fn()
		if err == nil {
			return nil
		}
		var netErr *netError
		if !transient(err) && !errors.As(err, &netErr) {
			return err
		}
		attempt++
		if attempt >= maxAttempts {
			return err
		}
		// Exponential backoff with jitter
		delay := time.Duration(float64(baseDelay) * math.Pow(2, float64(attempt-1)))
		if delay > maxDelay {
			delay = maxDelay
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay + time.Duration(float64(delay)*0.1)):
		}
	}
}

// netError is used only for errors.As matching when needed.
type netError struct{ error }

func transient(err error) bool {
	// Very basic heuristic: HTTP status already handled; here we can extend if needed.
	// Keep simple for v1.
	_ = err
	return true
}
