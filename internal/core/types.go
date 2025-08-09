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

type ToolDef struct {
	Name        string
	Description string
	JSONSchema  string
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
	Name string
	Args json.RawMessage
}
