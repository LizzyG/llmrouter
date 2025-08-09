package util

import "testing"

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
	if !contains(schema, "name") || !contains(schema, "age") {
		t.Fatalf("schema missing fields: %s", schema)
	}
}

func contains(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (find(haystack, needle) >= 0)
}

func find(h, n string) int {
	for i := 0; i+len(n) <= len(h); i++ {
		if h[i:i+len(n)] == n {
			return i
		}
	}
	return -1
}
