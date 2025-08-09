package llmrouter

import (
	"context"
	"encoding/json"
	"time"

	moderr "github.com/lizzyg/llmrouter/errors"
	"github.com/lizzyg/llmrouter/internal/util"
)

// Tool is implemented by any callable function the model can invoke.
// Parameters must return a pointer to a zero-value struct for JSON schema generation and unmarshalling.
type Tool interface {
	Name() string
	Description() string
	Parameters() any
	Execute(ctx context.Context, args any) (any, error)
}

// Client is the only type applications use.
// ExecuteRaw returns the final model content as a JSON string after the tool loop.
type Client interface {
	ExecuteRaw(ctx context.Context, req Request) (string, error)
}

// Execute executes the request through the tool loop and parses the final JSON into T.
// If T is string, the raw text is returned.
func Execute[T any](ctx context.Context, c Client, req Request) (T, error) {
	var zero T
	// If the client can use an output schema, leverage it for stricter results.
	type schemaExec interface {
		executeWithSchema(ctx context.Context, req Request, outputSchema string, requireStructured bool) (string, error)
	}
	if se, ok := c.(schemaExec); ok {
		// Build schema for T
		var zeroPtr *T
		schema := util.GenerateJSONSchema(zeroPtr)
		s, err := se.executeWithSchema(ctx, req, schema, true)
		if err != nil {
			return zero, err
		}
		if util.IsStringType[T]() {
			anyVal := any(s)
			return anyVal.(T), nil
		}
		var out T
		if err := json.Unmarshal([]byte(s), &out); err != nil {
			if repaired, ok := util.RepairJSON(s); ok {
				if err2 := json.Unmarshal([]byte(repaired), &out); err2 == nil {
					return out, nil
				}
			}
			return zero, moderr.ErrStructuredOutput
		}
		return out, nil
	}

	s, err := c.ExecuteRaw(ctx, req)
	if err != nil {
		return zero, err
	}
	if util.IsStringType[T]() {
		anyVal := any(s)
		return anyVal.(T), nil
	}
	var out T
	if err := json.Unmarshal([]byte(s), &out); err != nil {
		if repaired, ok := util.RepairJSON(s); ok {
			if err2 := json.Unmarshal([]byte(repaired), &out); err2 == nil {
				return out, nil
			}
		}
		return zero, moderr.ErrStructuredOutput
	}
	return out, nil
}

// Request describes a single LLM request.
type Request struct {
	Model          string
	Messages       []Message
	AllowWebSearch bool
	Tools          []Tool
	MaxTokens      int
	Temperature    float32
	TopP           float32

	// Optional overrides
	Timeout time.Duration
}

// Message is one conversational message.
type Message struct {
	Role    MessageRole
	Content string
	Images  []string // image URLs supported in v1
}

// MessageRole defines who authored a message.
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
)
