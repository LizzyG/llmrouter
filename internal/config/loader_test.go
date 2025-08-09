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
