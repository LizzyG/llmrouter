package config

import "sync"

// ResetForTest clears the cached config loader state.
// Intended for test code to re-load different configs within the same process.
func ResetForTest() {
	loaded = nil
	loadErr = nil
	loadOnce = sync.Once{}
}
