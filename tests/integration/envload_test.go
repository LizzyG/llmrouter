//go:build integration
// +build integration

package integration

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// init attempts to load a project-root .env so integration tests can pick up API keys
// without requiring shell export. Existing env vars are not overwritten.
func init() {
	tryLoadDotEnv()
}

func tryLoadDotEnv() {
	// Probe a few likely locations relative to this package directory
	candidates := []string{
		".env",
		filepath.Join("..", ".env"),
		filepath.Join("..", "..", ".env"),
		filepath.Join("..", "..", "..", ".env"),
	}
	var f *os.File
	var err error
	for _, p := range candidates {
		f, err = os.Open(p)
		if err == nil {
			defer f.Close()
			scanAndSetEnv(f)
			return
		}
	}
}

func scanAndSetEnv(f *os.File) {
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		val = strings.Trim(val, "\"'")
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, val)
		}
	}
}
