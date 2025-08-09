package main

import (
	"context"
	"fmt"

	llm "github.com/lizzyg/llmrouter"
)

type Answer struct {
	Summary string `json:"summary"`
}

func main() {
	client, err := llm.NewFromFile()
	if err != nil {
		panic(err)
	}
	ctx := context.Background()
	res, err := llm.Execute[Answer](ctx, client, llm.Request{
		Model: "gpt4o",
		Messages: []llm.Message{{
			Role:    llm.RoleUser,
			Content: "Return JSON {\"summary\":\"Hello\"}",
		}},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Summary)
}
