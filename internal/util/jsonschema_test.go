package util

import (
	"strings"
	"testing"
)

type sample struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestGenerateJSONSchema(t *testing.T) {
	s := &sample{}
	schema := GenerateJSONSchema(s)
	if len(schema) == 0 {
		t.Fatal("empty schema")
	}
	if !strings.Contains(schema, `"name"`) || !strings.Contains(schema, `"age"`) {
		t.Fatalf("schema missing fields: %s", schema)
	}
}
