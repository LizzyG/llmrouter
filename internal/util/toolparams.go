package util

import (
	"reflect"
	"strings"
)

// GenerateToolParameters generates a slice of core.ToolParameter directly from a struct type
// using reflection, avoiding the JSON marshaling/unmarshaling cycle.
func GenerateToolParameters(paramStruct any) ([]map[string]any, error) {
	if paramStruct == nil {
		return []map[string]any{}, nil
	}

	// Get the type of the parameter struct
	t := reflect.TypeOf(paramStruct)
	
	// Handle pointer types
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	
	// Must be a struct
	if t.Kind() != reflect.Struct {
		return []map[string]any{}, nil
	}

	params := make([]map[string]any, 0, t.NumField())
	
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		
		// Skip unexported fields
		if !field.IsExported() {
			continue
		}
		
		// Get field name (respect json tag if present)
		name := field.Name
		if jsonTag := field.Tag.Get("json"); jsonTag != "" {
			if parts := strings.Split(jsonTag, ","); len(parts) > 0 && parts[0] != "" {
				name = parts[0]
			}
		}
		
		// Skip fields with json:"-" tag
		if jsonTag := field.Tag.Get("json"); jsonTag == "-" {
			continue
		}
		
		// Determine if field is required (not a pointer, not zero value, or has required tag)
		required := true
		if field.Type.Kind() == reflect.Ptr {
			required = false
		}
		
		// Check for required tag
		if requiredTag := field.Tag.Get("required"); requiredTag == "false" {
			required = false
		} else if requiredTag == "true" {
			required = true
		}
		
		// Get description from tag
		description := field.Tag.Get("description")
		
		// Generate schema for the field type
		schema := generateSchemaForType(field.Type)
		
		param := map[string]any{
			"name":        name,
			"required":    required,
			"description": description,
			"schema":      schema,
		}
		

		
		params = append(params, param)
	}
	
	return params, nil
}

// generateSchemaForType generates a JSON schema fragment for a given Go type
func generateSchemaForType(t reflect.Type) map[string]any {
	schema := make(map[string]any)
	
	// Handle pointer types
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
		schema["nullable"] = true
	}
	
	switch t.Kind() {
	case reflect.String:
		schema["type"] = "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		schema["type"] = "integer"
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		schema["type"] = "integer"
		schema["minimum"] = 0
	case reflect.Float32, reflect.Float64:
		schema["type"] = "number"
	case reflect.Bool:
		schema["type"] = "boolean"
	case reflect.Array, reflect.Slice:
		schema["type"] = "array"
		if t.Elem().Kind() != reflect.Interface {
			schema["items"] = generateSchemaForType(t.Elem())
		}
	case reflect.Map:
		schema["type"] = "object"
		if t.Key().Kind() == reflect.String {
			schema["additionalProperties"] = true
		}
	case reflect.Struct:
		// For structs, we could recursively generate properties
		// For now, just mark as object type
		schema["type"] = "object"
		schema["additionalProperties"] = true
	case reflect.Interface:
		// Interface types can be anything
		schema["type"] = "object"
		schema["additionalProperties"] = true
	default:
		// Fallback for unknown types
		schema["type"] = "string"
	}
	
	return schema
}

// GenerateToolParametersWithRequired generates ToolParameters with explicit required field handling
// This is useful when you want to control which fields are required independently of their Go types
func GenerateToolParametersWithRequired(paramStruct any, requiredFields []string) ([]map[string]any, error) {
	params, err := GenerateToolParameters(paramStruct)
	if err != nil {
		return nil, err
	}
	
	// Create a set of required field names for efficient lookup
	requiredSet := make(map[string]bool)
	for _, field := range requiredFields {
		requiredSet[field] = true
	}
	
	// Update the required field for each parameter
	for _, param := range params {
		if name, ok := param["name"].(string); ok {
			param["required"] = requiredSet[name]
		}
	}
	
	return params, nil
}
