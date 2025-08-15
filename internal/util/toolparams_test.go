package util

import (
	"reflect"
	"testing"
)

// Test structs for testing
type SimpleParams struct {
	Name     string `json:"name" description:"The name of the item"`
	Age      int    `json:"age" description:"The age of the person"`
	IsActive bool   `json:"is_active" description:"Whether the item is active"`
}

type OptionalParams struct {
	Required   string  `json:"required" required:"true"`
	Optional   *string `json:"optional" description:"Optional field"`
	Hidden     string  `json:"-"`
	NoJSONTag  string  `description:"Field without json tag"`
	Complex    []int   `json:"complex" description:"Array of integers"`
}

type NestedParams struct {
	Simple SimpleParams `json:"simple" description:"Nested simple params"`
	Map    map[string]any `json:"map" description:"Map of values"`
}

func TestGenerateToolParameters(t *testing.T) {
	t.Run("simple_params", func(t *testing.T) {
		params, err := GenerateToolParameters(SimpleParams{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		
		if len(params) != 3 {
			t.Fatalf("expected 3 parameters, got %d", len(params))
		}
		
		// Check name field
		nameParam := findParamByName(params, "name")
		if nameParam == nil {
			t.Fatal("name parameter not found")
		}
		if nameParam["schema"].(map[string]any)["type"] != "string" {
			t.Errorf("expected type string for name, got %v", nameParam["schema"].(map[string]any)["type"])
		}
		if nameParam["required"] != true {
			t.Errorf("expected name to be required, got %v", nameParam["required"])
		}
		if nameParam["description"] != "The name of the item" {
			t.Errorf("expected description 'The name of the item', got %v", nameParam["description"])
		}
		
		// Check age field
		ageParam := findParamByName(params, "age")
		if ageParam == nil {
			t.Fatal("age parameter not found")
		}
		if ageParam["schema"].(map[string]any)["type"] != "integer" {
			t.Errorf("expected type integer for age, got %v", ageParam["schema"].(map[string]any)["type"])
		}
		
		// Check is_active field
		activeParam := findParamByName(params, "is_active")
		if activeParam == nil {
			t.Fatal("is_active parameter not found")
		}
		if activeParam["schema"].(map[string]any)["type"] != "boolean" {
			t.Errorf("expected type boolean for is_active, got %v", activeParam["schema"].(map[string]any)["type"])
		}
	})
	
	t.Run("optional_params", func(t *testing.T) {
		params, err := GenerateToolParameters(OptionalParams{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		
		if len(params) != 4 {
			t.Fatalf("expected 4 parameters, got %d", len(params))
		}
		
		// Check required field
		requiredParam := findParamByName(params, "required")
		if requiredParam == nil {
			t.Fatal("required parameter not found")
		}
		if requiredParam["required"] != true {
			t.Errorf("expected required to be true, got %v", requiredParam["required"])
		}
		
		// Check optional field (pointer type)
		optionalParam := findParamByName(params, "optional")
		if optionalParam == nil {
			t.Fatal("optional parameter not found")
		}
		if optionalParam["required"] != false {
			t.Errorf("expected optional to be false, got %v", optionalParam["required"])
		}
		if optionalParam["schema"].(map[string]any)["nullable"] != true {
			t.Errorf("expected optional to be nullable, got %v", optionalParam["schema"].(map[string]any)["nullable"])
		}
		
		// Check hidden field (should not be present)
		hiddenParam := findParamByName(params, "hidden")
		if hiddenParam != nil {
			t.Error("hidden parameter should not be present")
		}
		
		// Check no_json_tag field
		noJSONParam := findParamByName(params, "NoJSONTag")
		if noJSONParam == nil {
			t.Fatal("NoJSONTag parameter not found")
		}
		if noJSONParam["name"] != "NoJSONTag" {
			t.Errorf("expected name NoJSONTag, got %v", noJSONParam["name"])
		}
		
		// Check complex field (slice)
		complexParam := findParamByName(params, "complex")
		if complexParam == nil {
			t.Fatal("complex parameter not found")
		}
		if complexParam["schema"].(map[string]any)["type"] != "array" {
			t.Errorf("expected type array for complex, got %v", complexParam["schema"].(map[string]any)["type"])
		}
	})
	
	t.Run("nested_params", func(t *testing.T) {
		params, err := GenerateToolParameters(NestedParams{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		
		if len(params) != 2 {
			t.Fatalf("expected 2 parameters, got %d", len(params))
		}
		
		// Check simple field (struct)
		simpleParam := findParamByName(params, "simple")
		if simpleParam == nil {
			t.Fatal("simple parameter not found")
		}
		if simpleParam["schema"].(map[string]any)["type"] != "object" {
			t.Errorf("expected type object for simple, got %v", simpleParam["schema"].(map[string]any)["type"])
		}
		
		// Check map field
		mapParam := findParamByName(params, "map")
		if mapParam == nil {
			t.Fatal("map parameter not found")
		}
		if mapParam["schema"].(map[string]any)["type"] != "object" {
			t.Errorf("expected type object for map, got %v", mapParam["schema"].(map[string]any)["type"])
		}
	})
	
	t.Run("nil_input", func(t *testing.T) {
		params, err := GenerateToolParameters(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(params) != 0 {
			t.Errorf("expected 0 parameters for nil input, got %d", len(params))
		}
	})
	
	t.Run("non_struct_input", func(t *testing.T) {
		params, err := GenerateToolParameters("not a struct")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(params) != 0 {
			t.Errorf("expected 0 parameters for non-struct input, got %d", len(params))
		}
	})
	
	t.Run("pointer_to_struct", func(t *testing.T) {
		simple := &SimpleParams{}
		params, err := GenerateToolParameters(simple)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(params) != 3 {
			t.Fatalf("expected 3 parameters for pointer to struct, got %d", len(params))
		}
	})
}

func TestGenerateToolParametersWithRequired(t *testing.T) {
	requiredFields := []string{"name", "age"}
	
	params, err := GenerateToolParametersWithRequired(SimpleParams{}, requiredFields)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	
	if len(params) != 3 {
		t.Fatalf("expected 3 parameters, got %d", len(params))
	}
	
	// Check that name and age are required
	nameParam := findParamByName(params, "name")
	if nameParam == nil || nameParam["required"] != true {
		t.Error("name should be required")
	}
	
	ageParam := findParamByName(params, "age")
	if ageParam == nil || ageParam["required"] != true {
		t.Error("age should be required")
	}
	
	// Check that is_active is not required
	activeParam := findParamByName(params, "is_active")
	if activeParam == nil || activeParam["required"] != false {
		t.Error("is_active should not be required")
	}
}

func TestGenerateSchemaForType(t *testing.T) {
	t.Run("basic_types", func(t *testing.T) {
		tests := []struct {
			value    any
			expected string
		}{
			{"string", "string"},
			{42, "integer"},
			{3.14, "number"},
			{true, "boolean"},
		}
		
		for _, tt := range tests {
			t.Run(tt.expected, func(t *testing.T) {
				schema := generateSchemaForType(reflect.TypeOf(tt.value))
				if schema["type"] != tt.expected {
					t.Errorf("expected type %s, got %v", tt.expected, schema["type"])
				}
			})
		}
	})
	
	t.Run("pointer_types", func(t *testing.T) {
		var str *string
		schema := generateSchemaForType(reflect.TypeOf(str))
		
		if schema["type"] != "string" {
			t.Errorf("expected type string for pointer, got %v", schema["type"])
		}
		if schema["nullable"] != true {
			t.Errorf("expected nullable true for pointer, got %v", schema["nullable"])
		}
	})
	
	t.Run("slice_types", func(t *testing.T) {
		var ints []int
		schema := generateSchemaForType(reflect.TypeOf(ints))
		
		if schema["type"] != "array" {
			t.Errorf("expected type array for slice, got %v", schema["type"])
		}
		
		items, ok := schema["items"].(map[string]any)
		if !ok {
			t.Fatal("expected items schema for slice")
		}
		if items["type"] != "integer" {
			t.Errorf("expected items type integer, got %v", items["type"])
		}
	})
	
	t.Run("map_types", func(t *testing.T) {
		var m map[string]any
		schema := generateSchemaForType(reflect.TypeOf(m))
		
		if schema["type"] != "object" {
			t.Errorf("expected type object for map, got %v", schema["type"])
		}
		if schema["additionalProperties"] != true {
			t.Errorf("expected additionalProperties true for map, got %v", schema["additionalProperties"])
		}
	})
}

// Helper function to find a parameter by name
func findParamByName(params []map[string]any, name string) map[string]any {
	for _, param := range params {
		if param["name"] == name {
			return param
		}
	}
	return nil
}
