package providers

import (
	"log/slog"
	"net/http"

	moderr "github.com/lizzyg/llmrouter/errors"
	"github.com/lizzyg/llmrouter/internal/config"
	"github.com/lizzyg/llmrouter/internal/core"
	"github.com/lizzyg/llmrouter/internal/providers/gemini"
	"github.com/lizzyg/llmrouter/internal/providers/openai"
)

func NewProviderClient(mc config.ModelConfig, hc *http.Client, logger *slog.Logger) (core.RawClient, error) {
    switch mc.Provider {
	case "openai":
		return openai.New(mc, hc, logger), nil
	case "gemini":
		return gemini.New(mc, hc, logger), nil
	default:
        return nil, moderr.ErrUnknownProvider
	}
}
