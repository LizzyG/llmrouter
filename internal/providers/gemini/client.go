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
	GenerationConfig map[string]any   `json:"generationConfig,omitempty"`
}

type generateResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
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

	content := ""
	if len(gr.Candidates) > 0 && len(gr.Candidates[0].Content.Parts) > 0 {
		content = gr.Candidates[0].Content.Parts[0].Text
	}
	out := core.RawResponse{Content: content}
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
			parts = append(parts, map[string]any{"text": m.Content})
		}
		for _, img := range m.Images {
			parts = append(parts, map[string]any{
				"inline_data": map[string]any{
					"mime_type": "image/url",
					"data":      img,
				},
			})
		}
		out = append(out, map[string]any{
			"role":  m.Role,
			"parts": parts,
		})
	}
	return out
}

func mapTools(defs []core.ToolDef) []map[string]any {
	// Minimal placeholder; adjust when enabling Gemini tools formally
	out := make([]map[string]any, len(defs))
	for i, d := range defs {
		out[i] = map[string]any{
			"function_declarations": []map[string]any{
				{
					"name":        d.Name,
					"description": d.Description,
					"parameters":  json.RawMessage(d.JSONSchema),
				},
			},
		}
	}
	return out
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
