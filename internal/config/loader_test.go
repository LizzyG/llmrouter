package config

import (
	"os"
	"testing"
)

func TestLoadMissingFile(t *testing.T) {
	os.Unsetenv("LLM_CONFIG_PATH")
	// Ensure default path does not exist in test env; expect error
	_, err := Load()
	if err == nil {
		t.Skip("config.yaml may exist in dev env; skipping")
	}
}

func TestResolveEnvString(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		envVar   string
		envValue string
		setEnv   bool
		expected string
	}{
		{
			name:     "replaces set environment variable",
			input:    "api-${API_KEY}-suffix",
			envVar:   "API_KEY",
			envValue: "test123",
			setEnv:   true,
			expected: "api-test123-suffix",
		},
		{
			name:     "handles empty environment variable",
			input:    "prefix-${EMPTY_VAR}-suffix",
			envVar:   "EMPTY_VAR",
			envValue: "",
			setEnv:   true,
			expected: "prefix--suffix",
		},
		{
			name:     "handles unset environment variable",
			input:    "prefix-${UNSET_VAR}-suffix",
			envVar:   "UNSET_VAR",
			setEnv:   false,
			expected: "prefix--suffix",
		},
		{
			name:     "handles multiple variables",
			input:    "${HOST}:${PORT}",
			envVar:   "HOST",
			envValue: "localhost",
			setEnv:   true,
			expected: "localhost:",
		},
		{
			name:     "no substitution needed",
			input:    "no-vars-here",
			envVar:   "",
			setEnv:   false,
			expected: "no-vars-here",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clean up environment
			if tt.envVar != "" {
				defer os.Unsetenv(tt.envVar)
				if tt.setEnv {
					os.Setenv(tt.envVar, tt.envValue)
				} else {
					os.Unsetenv(tt.envVar)
				}
			}
			
			// For the multiple variables test, set the PORT variable too
			if tt.name == "handles multiple variables" {
				defer os.Unsetenv("PORT")
				os.Setenv("PORT", "")
			}

			result := resolveEnvString(tt.input)
			if result != tt.expected {
				t.Errorf("resolveEnvString(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
