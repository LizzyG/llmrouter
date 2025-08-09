package gemini

import (
	"bytes"
	"context"
	"encoding/json"
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
	return &Client{apiKey: mc.APIKey, httpClient: hc, logger: logger, model: mc.Model}
}

type generateRequest struct {
	Contents         []map[string]any `json:"contents"`
	Tools            []map[string]any `json:"tools,omitempty"`
	ToolConfig       map[string]any   `json:"toolConfig,omitempty"`
	GenerationConfig map[string]any   `json:"generationConfig,omitempty"`
}

type generateResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text         string         `json:"text"`
				FunctionCall map[string]any `json:"functionCall"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	Usage struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	// Tool calls mapping is provider-specific; placeholder for v1.
}

func (c *Client) Call(ctx context.Context, params core.CallParams) (core.RawResponse, error) {
	payload := generateRequest{
		Contents:         mapMessages(params.Messages),
		GenerationConfig: map[string]any{},
	}
	if params.MaxTokens > 0 {
		payload.GenerationConfig["maxOutputTokens"] = params.MaxTokens
	}
	if params.Temperature > 0 {
		payload.GenerationConfig["temperature"] = params.Temperature
	}
	if params.TopP > 0 {
		payload.GenerationConfig["topP"] = params.TopP
	}
	if len(params.ToolDefs) > 0 {
		payload.Tools = mapTools(params.ToolDefs)
		// Encourage the model to use at least one function
		payload.ToolConfig = map[string]any{
			"functionCallingConfig": map[string]any{
				"mode": "ANY",
			},
		}
	}
	if params.OutputSchema != "" {
		payload.GenerationConfig["responseMimeType"] = "application/json"
		payload.GenerationConfig["responseSchema"] = convertJSONSchemaToGemini(params.OutputSchema)
	}

	body, _ := json.Marshal(payload)
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", c.model, c.apiKey)

	var gr generateResponse
	err := withRetry(ctx, func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := c.httpClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			b, _ := io.ReadAll(resp.Body)
			return fmt.Errorf("gemini http %d: %s", resp.StatusCode, string(b))
		}
		dec := json.NewDecoder(resp.Body)
		return dec.Decode(&gr)
	})
	if err != nil {
		return core.RawResponse{}, err
	}

	out := core.RawResponse{}
	if len(gr.Candidates) > 0 && len(gr.Candidates[0].Content.Parts) > 0 {
		parts := gr.Candidates[0].Content.Parts
		// If model requested function calls, surface them
		toolCalls := make([]core.ToolCall, 0)
		for _, p := range parts {
			if fc, ok := p.FunctionCall["name"].(string); ok && fc != "" {
				// Extract args map
				var raw json.RawMessage
				if args, ok2 := p.FunctionCall["args"].(map[string]any); ok2 {
					b, _ := json.Marshal(args)
					raw = b
				} else if anyArgs, ok3 := p.FunctionCall["args"].(any); ok3 {
					b, _ := json.Marshal(anyArgs)
					raw = b
				}
				toolCalls = append(toolCalls, core.ToolCall{Name: fc, Args: raw})
			}
		}
		if len(toolCalls) > 0 {
			out.ToolCalls = toolCalls
		} else {
			// Aggregate text content
			acc := ""
			for _, p := range parts {
				if p.Text != "" {
					if acc == "" {
						acc = p.Text
					} else {
						acc += "\n" + p.Text
					}
				}
			}
			out.Content = acc
		}
	}
	out.Usage = core.Usage{
		PromptTokens:     gr.Usage.PromptTokenCount,
		CompletionTokens: gr.Usage.CandidatesTokenCount,
		TotalTokens:      gr.Usage.TotalTokenCount,
	}
	return out, nil
}

func mapMessages(msgs []core.Message) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		parts := []any{}
		if m.Content != "" {
			// Try to interpret assistant tool results (formatToolResult JSON) as functionResponse
			if m.Role == "assistant" {
				var obj map[string]any
				if json.Unmarshal([]byte(m.Content), &obj) == nil {
					if toolName, ok := obj["tool"].(string); ok {
						parts = append(parts, map[string]any{
							"functionResponse": map[string]any{
								"name":     toolName,
								"response": obj["result"],
							},
						})
					} else {
						parts = append(parts, map[string]any{"text": m.Content})
					}
				} else {
					parts = append(parts, map[string]any{"text": m.Content})
				}
			} else {
				parts = append(parts, map[string]any{"text": m.Content})
			}
		}
		for _, img := range m.Images {
			parts = append(parts, map[string]any{
				"inline_data": map[string]any{
					"mime_type": "image/url",
					"data":      img,
				},
			})
		}
		role := m.Role
		// When sending functionResponse, Gemini expects role "tool"
		if len(parts) > 0 {
			if mr, ok := parts[0].(map[string]any); ok {
				if _, hasFR := mr["functionResponse"]; hasFR {
					role = "tool"
				}
			}
		}
		out = append(out, map[string]any{"role": role, "parts": parts})
	}
	return out
}

func mapTools(defs []core.ToolDef) []map[string]any {
	out := make([]map[string]any, len(defs))
	for i, d := range defs {
		params := convertJSONSchemaToGemini(d.JSONSchema)
		out[i] = map[string]any{
			"function_declarations": []map[string]any{
				{
					"name":        d.Name,
					"description": d.Description,
					"parameters":  params,
				},
			},
		}
	}
	return out
}

// convertJSONSchemaToGemini transforms a standard JSON Schema into Gemini's schema dialect.
// Gemini expects:
// { type: OBJECT|ARRAY|STRING|INTEGER|NUMBER|BOOLEAN, properties?: {...}, items?: {...} }
func convertJSONSchemaToGemini(schema string) any {
	var m map[string]any
	if err := json.Unmarshal([]byte(schema), &m); err != nil {
		return map[string]any{"type": "OBJECT", "properties": map[string]any{}}
	}
	return toGeminiSchema(m)
}

func toGeminiSchema(node map[string]any) map[string]any {
	// Strip meta keys
	delete(node, "$schema")
	delete(node, "$id")
	delete(node, "$defs")
	delete(node, "definitions")
	delete(node, "$ref")
	delete(node, "title")
	delete(node, "description")
	delete(node, "additionalProperties")

	t, _ := node["type"].(string)
	switch t {
	case "string":
		return map[string]any{"type": "STRING"}
	case "integer":
		return map[string]any{"type": "INTEGER"}
	case "number":
		return map[string]any{"type": "NUMBER"}
	case "boolean":
		return map[string]any{"type": "BOOLEAN"}
	case "array":
		items := map[string]any{}
		if it, ok := node["items"].(map[string]any); ok {
			items = toGeminiSchema(it)
		}
		return map[string]any{"type": "ARRAY", "items": items}
	case "object", "OBJECT", "":
		propsOut := map[string]any{}
		if props, ok := node["properties"].(map[string]any); ok {
			for k, v := range props {
				if child, ok2 := v.(map[string]any); ok2 {
					propsOut[k] = toGeminiSchema(child)
				}
			}
		}
		return map[string]any{"type": "OBJECT", "properties": propsOut}
	default:
		return map[string]any{"type": "OBJECT", "properties": map[string]any{}}
	}
}

func withRetry(ctx context.Context, fn func() error) error {
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
		attempt++
		if attempt >= maxAttempts {
			return err
		}
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
