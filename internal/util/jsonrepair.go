package util

import (
	"strings"
)

// RepairJSON attempts minimal fixups to coerce a model response into valid JSON.
// - Strips markdown code fences
// - Trims whitespace
// - Attempts to extract the outermost JSON object or array
// Returns the possibly repaired string and true if modified.
func RepairJSON(s string) (string, bool) {
	original := s
	s = strings.TrimSpace(s)

	// Strip ```json ... ``` or ``` ... ``` fences
	if strings.HasPrefix(s, "```") && strings.HasSuffix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
		if strings.HasPrefix(strings.ToLower(s), "json") {
			s = strings.TrimSpace(s[4:])
		}
	}

	// Try to trim to first '{' or '[' and matching closing '}' or ']'
	// Simple heuristic; avoid heavy parsing to keep it lightweight.
	idxObj := strings.IndexByte(s, '{')
	idxArr := strings.IndexByte(s, '[')
	start := -1
	if idxObj >= 0 && (idxArr < 0 || idxObj < idxArr) {
		start = idxObj
	} else if idxArr >= 0 {
		start = idxArr
	}
	if start >= 0 {
		s = s[start:]
		// Attempt to cut trailing content after last matching closing brace/bracket
		// This is a conservative trim: take up to last '}' or ']'.
		lastObj := strings.LastIndexByte(s, '}')
		lastArr := strings.LastIndexByte(s, ']')
		end := -1
		if lastObj >= 0 && (lastArr < 0 || lastObj > lastArr) {
			end = lastObj + 1
		} else if lastArr >= 0 {
			end = lastArr + 1
		}
		if end > 0 && end <= len(s) {
			s = s[:end]
		}
	}

	return s, s != original
}
