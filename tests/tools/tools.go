package tools

import (
	"context"

	llm "github.com/lizzyg/llmrouter"
)

// GetUserLocation
type GetUserLocationArgs struct{}

type GetUserLocationTool struct{}

func (t *GetUserLocationTool) Name() string { return "GetUserLocation" }
func (t *GetUserLocationTool) Description() string {
	return "Returns the user's current city and state"
}
func (t *GetUserLocationTool) Parameters() any { return &GetUserLocationArgs{} }
func (t *GetUserLocationTool) Execute(ctx context.Context, args any) (any, error) {
	return map[string]any{"location": "Portland, Oregon"}, nil
}

// GetWeatherInLocation
type GetWeatherArgs struct {
	Location string `json:"location"`
}

type GetWeatherInLocationTool struct{}

func (t *GetWeatherInLocationTool) Name() string { return "GetWeatherInLocation" }
func (t *GetWeatherInLocationTool) Description() string {
	return "Returns current weather for a location"
}
func (t *GetWeatherInLocationTool) Parameters() any { return &GetWeatherArgs{} }
func (t *GetWeatherInLocationTool) Execute(ctx context.Context, args any) (any, error) {
	a := args.(*GetWeatherArgs)
	return map[string]any{"weather": "Sunny and mild in " + a.Location}, nil
}

// Helper to return both tools in correct order
func LocationWeatherTools() []llm.Tool {
	return []llm.Tool{&GetUserLocationTool{}, &GetWeatherInLocationTool{}}
}
