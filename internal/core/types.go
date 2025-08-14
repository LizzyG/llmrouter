package core

import (
	"context"
	"encoding/json"
)

// RawClient is implemented by provider adapters.
type RawClient interface {
	Call(ctx context.Context, params CallParams) (RawResponse, error)
}

type CallParams struct {
	Model        string
	Messages     []Message
	ToolDefs     []ToolDef
	OutputSchema string
	MaxTokens    int
	Temperature  float32
	TopP         float32
}

type Message struct {
	Role    string
	Content string
	Images  []string
}

// ToolParameter represents a single parameter accepted by a tool.
// Schema should contain a JSON Schema fragment describing this parameter's type.
type ToolParameter struct {
	Name        string         `json:"name"`
	Required    bool           `json:"required"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
}

// ToolDef describes a tool and its parameters in a provider-agnostic form.
// Providers can derive their own JSON schema or function declaration format from it.
type ToolDef struct {
	Name        string
	Description string
	Parameters  []ToolParameter
}

type RawResponse struct {
	Content   string
	ToolCalls []ToolCall
	Usage     Usage
}

type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

type ToolCall struct {
	CallID string
	Name   string
	Args   json.RawMessage
}

// GenerateJSONSchemaFromToolDef produces a standard JSON Schema object string
// of the shape {"type":"object","properties":{...},"required":[...]}
// using the ToolDef.Parameters list.
func GenerateJSONSchemaFromToolDef(def ToolDef) string {
	props := map[string]any{}
	required := make([]string, 0)
	for _, p := range def.Parameters {
		if p.Schema == nil {
			p.Schema = map[string]any{"type": "string"}
		}
		props[p.Name] = p.Schema
		if p.Required {
			required = append(required, p.Name)
		}
	}
	schema := map[string]any{
		"type":       "object",
		"properties": props,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	b, err := json.Marshal(schema)
	if err != nil {
		// Return a valid empty object schema on marshal failure
		return "{}"
	}
	return string(b)
}
