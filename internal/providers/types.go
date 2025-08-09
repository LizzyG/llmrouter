package providers

import "encoding/json"

// RawClient is implemented by provider adapters.
type RawClient interface {
	Call(ctx Context, params CallParams) (RawResponse, error)
}

// Context is a narrowed interface of context.Context to avoid importing it everywhere.
type Context interface {
	Done() <-chan struct{}
	Err() error
	Deadline() (deadline Time, ok bool)
}

// Time is an alias to avoid bringing time into this package's public API.
type Time = interface{}

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
