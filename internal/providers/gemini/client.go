package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"os"
	"strings"
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
	Contents          []map[string]any `json:"contents"`
	Tools             []map[string]any `json:"tools,omitempty"`
	ToolConfig        map[string]any   `json:"toolConfig,omitempty"`
	GenerationConfig  map[string]any   `json:"generationConfig,omitempty"`
	SystemInstruction map[string]any   `json:"systemInstruction,omitempty"`
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
	// Split out system messages to use systemInstruction per Gemini API
	sysMsgs := make([]core.Message, 0)
	nonSys := make([]core.Message, 0, len(params.Messages))
	for _, m := range params.Messages {
		if m.Role == "system" {
			sysMsgs = append(sysMsgs, m)
			continue
		}
		nonSys = append(nonSys, m)
	}

	payload := generateRequest{
		Contents:          mapMessages(nonSys),
		GenerationConfig:  map[string]any{},
		SystemInstruction: nil,
	}
	if len(sysMsgs) > 0 {
		parts := make([]map[string]any, 0, len(sysMsgs))
		for _, sm := range sysMsgs {
			if sm.Content != "" {
				parts = append(parts, map[string]any{"text": sm.Content})
			}
		}
		if len(parts) > 0 {
			payload.SystemInstruction = map[string]any{"parts": parts}
		}
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
		// When tools are present, don't use structured output as it conflicts with tool calling.
		// Heuristic: if we've already provided tool responses in the conversation, allow the model
		// to choose whether to call again or finalize by using AUTO. Otherwise, use ANY to encourage
		// an initial tool call.
		mode := "ANY"
		for _, c := range payload.Contents {
			if parts, ok := c["parts"].([]any); ok {
				for _, p := range parts {
					if pm, ok2 := p.(map[string]any); ok2 {
						if _, hasFR := pm["functionResponse"]; hasFR {
							mode = "AUTO"
							break
						}
					}
				}
			}
			if mode == "AUTO" {
				break
			}
		}
		payload.ToolConfig = map[string]any{
			"functionCallingConfig": map[string]any{"mode": mode},
		}
	} else if params.OutputSchema != "" {
		// Only use structured output when no tools are present
		payload.GenerationConfig["responseMimeType"] = "application/json"
		// Convert the provided schema (already a JSON string) into Gemini dialect
		payload.GenerationConfig["responseSchema"] = convertJSONSchemaToGemini(params.OutputSchema)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return core.RawResponse{}, fmt.Errorf("gemini marshal payload: %w", err)
	}
	if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
		c.logger.Info("gemini request payload", "payload", string(body))
	}
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", c.model, c.apiKey)

	var gr generateResponse
	err = withRetry(ctx, func() error {
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

	if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
		respBody, _ := json.Marshal(gr)
		c.logger.Info("gemini response", "response", string(respBody))
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
				} else {
					if anyArgs := p.FunctionCall["args"]; anyArgs != nil {
						b, _ := json.Marshal(anyArgs)
						raw = b
					}
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
	if os.Getenv("LLM_VERBOSE_MESSAGES") == "1" {
		for i, m := range msgs {
			slog.Default().Info("gemini mapMessages debug",
				slog.Int("index", i),
				slog.String("role", m.Role),
				slog.String("content", m.Content),
			)
		}
	}
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		parts := []any{}
		// Prefer structured fields first
		if len(m.ToolCalls) > 0 {
			for _, it := range m.ToolCalls {
				var args any
				if len(it.Args) > 0 {
					_ = json.Unmarshal(it.Args, &args)
				}
				parts = append(parts, map[string]any{
					"functionCall": map[string]any{
						"name": it.Name,
						"args": args,
					},
				})
			}
			role := m.Role
			if role == "assistant" {
				role = "model"
			}
			out = append(out, map[string]any{"role": role, "parts": parts})
			continue
		}
		if len(m.ToolResults) > 0 {
			for _, tr := range m.ToolResults {
				parts = append(parts, map[string]any{
					"functionResponse": map[string]any{
						"name":     tr.Name,
						"response": tr.Result,
					},
				})
			}
			out = append(out, map[string]any{"role": "tool", "parts": parts})
			continue
		}
		if m.Content != "" {
			// Try to interpret assistant tool results (formatToolResult JSON) as functionResponse
			if m.Role == "assistant" {
				// Path A: assistant content contains function calls to be echoed (for pairing)
				var fcArray []map[string]any
				if err := json.Unmarshal([]byte(m.Content), &fcArray); err == nil {
					// If objects look like {"tool": name, "args": {...}}, emit functionCall parts
					valid := true
					partsFromFC := []any{}
					for _, item := range fcArray {
						name, okN := item["tool"].(string)
						args, okA := item["args"].(map[string]any)
						if okN && okA {
							partsFromFC = append(partsFromFC, map[string]any{
								"functionCall": map[string]any{
									"name": name,
									"args": args,
								},
							})
						} else {
							valid = false
							break
						}
					}
					if valid && len(partsFromFC) > 0 {
						parts = append(parts, partsFromFC...)
						// Note: role remains "assistant" when sending functionCall echo
						// Short-circuit remaining parsing for this message
						out = append(out, map[string]any{"role": m.Role, "parts": parts})
						continue
					}
				} else {
					slog.Default().Debug("gemini: failed to parse assistant content as functionCall array", "error", err)
				}

				// Path B: assistant contains tool results to send back as functionResponse
				parsed := false
				var obj map[string]any
				if err := json.Unmarshal([]byte(m.Content), &obj); err == nil {
					if toolName, ok := obj["tool"].(string); ok {
						parts = append(parts, map[string]any{
							"functionResponse": map[string]any{
								"name":     toolName,
								"response": obj["result"],
							},
						})
						parsed = true
					}
				} else {
					slog.Default().Debug("gemini: failed to parse assistant content as single tool result", "error", err)
				}
				// If not parsed as single object, try array of tool results
				if !parsed {
					var objArray []map[string]any
					if err := json.Unmarshal([]byte(m.Content), &objArray); err == nil {
						for _, toolResult := range objArray {
							if toolName, ok := toolResult["tool"].(string); ok {
								parts = append(parts, map[string]any{
									"functionResponse": map[string]any{
										"name":     toolName,
										"response": toolResult["result"],
									},
								})
								parsed = true
							}
						}
					} else {
						slog.Default().Debug("gemini: failed to parse assistant content as tool results array", "error", err)
					}
				}
				// Fallback to plain text if neither parse path matched
				if !parsed {
					parts = append(parts, map[string]any{"text": m.Content})
				}
			} else {
				parts = append(parts, map[string]any{"text": m.Content})
			}
		}
		for _, img := range m.Images {
			parts = append(parts, map[string]any{
				"fileData": map[string]any{
					"fileUri": img,
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
		// Map assistant to model as per Gemini's role names
		if role == "assistant" {
			role = "model"
		}
		out = append(out, map[string]any{"role": role, "parts": parts})
	}
	return out
}

func mapTools(defs []core.ToolDef) []map[string]any {
	// Gemini expects a single tools entry containing all functionDeclarations, with camelCase keys.
	// Shape: tools: [ { functionDeclarations: [ { name, description, parameters }, ... ] } ]
	fns := make([]map[string]any, 0, len(defs))
	for _, d := range defs {
		// Build JSON schema object from ToolDef.Parameters
		schema := core.GenerateJSONSchemaFromToolDef(d)
		params := convertJSONSchemaToGemini(schema)
		fns = append(fns, map[string]any{
			"name":        d.Name,
			"description": d.Description,
			"parameters":  params,
		})
	}
	if len(fns) == 0 {
		return nil
	}
	return []map[string]any{
		{
			"functionDeclarations": fns,
		},
	}
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
		// Only retry transient errors similar to OpenAI client behavior
		if !isTransient(err) {
			return err
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

// Borrow the OpenAI transient detection pattern for Gemini simple errors.
// Gemini uses plain errors; we retry on 429/5xx strings or network timeouts if provided.
func isTransient(err error) bool {
	// String sniffing for HTTP status codes in error text (since Gemini path uses fmt.Errorf)
	if err == nil {
		return false
	}
	es := err.Error()
	if strings.Contains(es, " http 429:") {
		return true
	}
	// Generic 5xx detection
	if strings.Contains(es, " http 5") { // e.g., "http 500:", "http 503:"
		return true
	}
	// Network timeouts
	var ne net.Error
	if errors.As(err, &ne) {
		if ne.Timeout() {
			return true
		}
	}
	return false
}
