package util

import "testing"

func TestRepairJSON(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{`{"a":1}`, `{"a":1}`},
		{"```json\n{\"a\":1}\n```", `{"a":1}`},
		{"garbage before {\"a\":1} trailing", `{"a":1}`},
		{"prefix [1,2,3] suffix", `[1,2,3]`},
	}
	for i, c := range cases {
		got, _ := RepairJSON(c.in)
		if got != c.want {
			t.Fatalf("case %d: want %q got %q", i, c.want, got)
		}
	}
}
