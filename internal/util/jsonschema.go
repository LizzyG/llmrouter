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
