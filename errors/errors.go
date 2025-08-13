package errors

import "errors"

var (
	ErrNoMatchingModel  = errors.New("no matching model found")
	ErrUnknownTool      = errors.New("unknown tool requested")
	ErrMaxToolTurns     = errors.New("max tool turns exceeded")
	ErrStructuredOutput = errors.New("structured output required but invalid")
    ErrUnknownProvider  = errors.New("unknown provider")
)
