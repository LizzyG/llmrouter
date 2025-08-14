package config

import (
	"os"
	"regexp"
	"strings"
	"sync"

	"github.com/knadh/koanf/parsers/yaml"
	kenv "github.com/knadh/koanf/providers/env"
	kfile "github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
)

// LLMConfig is the root config structure.
type LLMConfig struct {
	Models map[string]ModelConfig `koanf:"models"`
}

// ModelConfig defines a single model entry in config.
type ModelConfig struct {
	Provider                 string `koanf:"provider"`
	Model                    string `koanf:"model"`
	APIKey                   string `koanf:"api_key"`
    WebVariant               string `koanf:"web_variant"`
	SupportsWebSearch        bool   `koanf:"supports_web_search"`
	SupportsTools            bool   `koanf:"supports_tools"`
	SupportsStructuredOutput bool   `koanf:"supports_structured_output"`
	ContextWindow            int    `koanf:"context_window"`
	MaxOutputTokens          int    `koanf:"max_output_tokens"`
}

var (
	loadOnce sync.Once
	loaded   *LLMConfig
	loadErr  error
)

// Load loads configuration from path or default locations. Load is safe for repeated calls.
//
// Priority:
// 1. LLM_CONFIG_PATH if set
// 2. ./config.yaml
func Load() (*LLMConfig, error) {
	loadOnce.Do(func() {
		k := koanf.New(".")

		// Determine path
		path := os.Getenv("LLM_CONFIG_PATH")
		if path == "" {
			path = "config.yaml"
		}

		// Load file with YAML parser
		if err := k.Load(kfile.Provider(path), yaml.Parser()); err != nil {
			loadErr = err
			return
		}

		// Environment overrides: LLM__MODELS__gpt4o__api_key=...
		// Double underscore splits levels.
		if err := k.Load(kenv.Provider("LLM__", "__", func(s string) string {
			return strings.ToLower(strings.TrimPrefix(s, "LLM__"))
		}), nil); err != nil {
			loadErr = err
			return
		}

		var cfg LLMConfig
		if err := k.Unmarshal("llm", &cfg); err != nil {
			loadErr = err
			return
		}

		// Resolve environment variables in string fields
		resolveEnvVars(&cfg)

		loaded = &cfg
	})
	return loaded, loadErr
}

var envVarRegex = regexp.MustCompile(`\$\{([^}]+)\}`)

// resolveEnvVars resolves ${VAR} patterns in config string fields
func resolveEnvVars(cfg *LLMConfig) {
	for key, model := range cfg.Models {
		model.APIKey = resolveEnvString(model.APIKey)
		model.Provider = resolveEnvString(model.Provider)
		model.Model = resolveEnvString(model.Model)
		cfg.Models[key] = model
	}
}

// resolveEnvString replaces ${VAR} with environment variable values
func resolveEnvString(s string) string {
	return envVarRegex.ReplaceAllStringFunc(s, func(match string) string {
		// Extract variable name from ${VAR}
		varName := match[2 : len(match)-1] // Remove ${ and }
		if value := os.Getenv(varName); value != "" {
			return value
		}
		return match // Return original if env var not found
	})
}
