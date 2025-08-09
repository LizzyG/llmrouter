package util

import (
	"encoding/json"
	"reflect"

	"github.com/invopop/jsonschema"
)

// GenerateJSONSchema returns a JSON schema string for the given object type.
// The object should be a pointer to a struct to capture fields and tags.
func GenerateJSONSchema(obj any) string {
	r := new(jsonschema.Reflector)
	// Configure reflector if needed in the future.
	schema := r.Reflect(obj)
	b, _ := json.Marshal(schema)
	return string(b)
}

// IsStringType reports whether T is string for generics handling.
func IsStringType[T any]() bool {
	var zero T
	return reflect.TypeOf(zero) == reflect.TypeOf("")
}

// GenerateToolJSONSchema returns a provider-friendly function parameter schema.
// It sanitizes the reflected schema by removing meta keys that some providers reject
// (e.g., $schema, $id, $defs, $ref, title, description) and ensures a minimal
// object-type schema even for empty structs.
func GenerateToolJSONSchema(obj any) string {
	raw := GenerateJSONSchema(obj)
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		// Fall back to a minimal object
		return `{"type":"object","properties":{}}`
	}
	// Remove meta keys
	delete(m, "$schema")
	delete(m, "$id")
	delete(m, "$defs")
	delete(m, "$ref")
	delete(m, "title")
	delete(m, "description")
	// Ensure type object
	if _, ok := m["type"]; !ok {
		m["type"] = "object"
	}
	if m["type"] != "object" {
		m["type"] = "object"
	}
	// Ensure properties map exists
	if _, ok := m["properties"]; !ok {
		m["properties"] = map[string]any{}
	}
	b, _ := json.Marshal(m)
	return string(b)
}

// GenerateResponseJSONSchema attempts to inline top-level $ref to a definition so that
// providers that don't support $ref can still receive a concrete object schema.
// Falls back to a minimal object schema on failure.
func GenerateResponseJSONSchema(obj any) string {
	raw := GenerateJSONSchema(obj)
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return `{"type":"object","properties":{}}`
	}
	// Try to inline $ref
	if ref, ok := m["$ref"].(string); ok && ref != "" {
		// Supported pattern: "#/$defs/TypeName" or "#/definitions/TypeName"
		var defsKey string
		if _, ok := m["$defs"]; ok {
			defsKey = "$defs"
		} else if _, ok := m["definitions"]; ok {
			defsKey = "definitions"
		}
		if defsKey != "" {
			if defs, ok := m[defsKey].(map[string]any); ok {
				// Extract name after last '/'
				lastSlash := -1
				for i := len(ref) - 1; i >= 0; i-- {
					if ref[i] == '/' {
						lastSlash = i
						break
					}
				}
				if lastSlash >= 0 && lastSlash+1 < len(ref) {
					name := ref[lastSlash+1:]
					if target, ok := defs[name].(map[string]any); ok {
						// Replace root with target schema
						for k := range m {
							delete(m, k)
						}
						for k, v := range target {
							m[k] = v
						}
					}
				}
			}
		}
	}
	// Remove meta keys
	delete(m, "$schema")
	delete(m, "$id")
	delete(m, "$defs")
	delete(m, "definitions")
	delete(m, "$ref")
	delete(m, "title")
	delete(m, "description")
	if t, ok := m["type"].(string); !ok || t == "" || t == "null" {
		m["type"] = "object"
	}
	if m["type"] != "object" {
		m["type"] = "object"
	}
	if _, ok := m["properties"]; !ok {
		m["properties"] = map[string]any{}
	}
	b, _ := json.Marshal(m)
	return string(b)
}

// SanitizeResponseSchemaJSON accepts a raw JSON schema string and applies the same
// sanitization/inlining rules as GenerateResponseJSONSchema.
func SanitizeResponseSchemaJSON(schemaStr string) string {
	var m map[string]any
	if err := json.Unmarshal([]byte(schemaStr), &m); err != nil {
		return `{"type":"object","properties":{}}`
	}
	// Inline $ref if present
	if ref, ok := m["$ref"].(string); ok && ref != "" {
		var defsKey string
		if _, ok := m["$defs"]; ok {
			defsKey = "$defs"
		} else if _, ok := m["definitions"]; ok {
			defsKey = "definitions"
		}
		if defsKey != "" {
			if defs, ok := m[defsKey].(map[string]any); ok {
				lastSlash := -1
				for i := len(ref) - 1; i >= 0; i-- {
					if ref[i] == '/' {
						lastSlash = i
						break
					}
				}
				if lastSlash >= 0 && lastSlash+1 < len(ref) {
					name := ref[lastSlash+1:]
					if target, ok := defs[name].(map[string]any); ok {
						for k := range m {
							delete(m, k)
						}
						for k, v := range target {
							m[k] = v
						}
					}
				}
			}
		}
	}
	delete(m, "$schema")
	delete(m, "$id")
	delete(m, "$defs")
	delete(m, "definitions")
	delete(m, "$ref")
	delete(m, "title")
	delete(m, "description")
	if t, ok := m["type"].(string); !ok || t == "" || t == "null" {
		m["type"] = "object"
	}
	if m["type"] != "object" {
		m["type"] = "object"
	}
	if _, ok := m["properties"]; !ok {
		m["properties"] = map[string]any{}
	}
	b, _ := json.Marshal(m)
	return string(b)
}
