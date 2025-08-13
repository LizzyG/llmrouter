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

	inlineTopLevelRef(m)
	sanitizeMetaAndCoerceObject(m, true)
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

	inlineTopLevelRef(m)
	sanitizeMetaAndCoerceObject(m, false)
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

	inlineTopLevelRef(m)
	sanitizeMetaAndCoerceObject(m, false)
	b, _ := json.Marshal(m)
	return string(b)
}

// inlineTopLevelRef attempts to inline a top-level $ref pointing to a definition under $defs or definitions.
// It mutates the provided map in place, replacing the root with the referenced schema when possible.
func inlineTopLevelRef(m map[string]any) {
	ref, ok := m["$ref"].(string)
	if !ok || ref == "" {
		return
	}
	var defsKey string
	if _, ok := m["$defs"]; ok {
		defsKey = "$defs"
	} else if _, ok := m["definitions"]; ok {
		defsKey = "definitions"
	}
	if defsKey == "" {
		return
	}
	defs, ok := m[defsKey].(map[string]any)
	if !ok {
		return
	}
	lastSlash := -1
	for i := len(ref) - 1; i >= 0; i-- {
		if ref[i] == '/' {
			lastSlash = i
			break
		}
	}
	if lastSlash < 0 || lastSlash+1 >= len(ref) {
		return
	}
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

// sanitizeMetaAndCoerceObject removes schema meta keys providers may reject and ensures
// the root is a concrete object with a properties map. If dropAdditionalProps is true,
// it also removes additionalProperties.
func sanitizeMetaAndCoerceObject(m map[string]any, dropAdditionalProps bool) {
	delete(m, "$schema")
	delete(m, "$id")
	delete(m, "$defs")
	delete(m, "definitions")
	delete(m, "$ref")
	delete(m, "title")
	delete(m, "description")
	if dropAdditionalProps {
		delete(m, "additionalProperties")
	}
	if t, ok := m["type"].(string); !ok || t == "" || t == "null" {
		m["type"] = "object"
	}
	if m["type"] != "object" {
		m["type"] = "object"
	}
	if _, ok := m["properties"]; !ok {
		m["properties"] = map[string]any{}
	}
}
