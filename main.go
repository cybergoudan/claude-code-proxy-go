package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// This program exposes an Anthropic Messages API compatible endpoint (/v1/messages)
// and forwards to an OpenAI-compatible Chat Completions upstream (/v1/chat/completions).
// It is designed to let Claude Code use your existing upstream (base URL + key)
// while keeping Claude Code's local tools (bash/git/files/MCP) fully functional.

func main() {
	addr := getenv("LISTEN", ":8088")
	upBase := strings.TrimRight(mustGetenv("UPSTREAM_BASE_URL"), "/")
	upKey := mustGetenv("UPSTREAM_API_KEY")
	upModel := getenv("UPSTREAM_MODEL", "gpt-5.2")

	// Validate URL early
	if _, err := url.Parse(upBase); err != nil {
		log.Fatalf("invalid UPSTREAM_BASE_URL: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		handleMessages(w, r, upBase, upKey, upModel)
	})

	// Optional minimal models endpoint
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []any{
				map[string]any{"id": upModel, "type": "model"},
			},
			"object": "list",
		})
	})

	srv := &http.Server{
		Addr:              addr,
		Handler:           logging(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("listening on %s", addr)
	log.Printf("upstream=%s model=%s", upBase, upModel)
	log.Fatal(srv.ListenAndServe())
}

func handleMessages(w http.ResponseWriter, r *http.Request, upstreamBase, upstreamKey, upstreamModel string) {
	ctx := r.Context()

	var req anthropicMessagesRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 10<<20)).Decode(&req); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Invalid JSON body")
		return
	}

	// Force model to configured upstream model (as requested)
	_ = req.Model
	
	// Convert to upstream OpenAI chat.completions request
	upReq, err := toUpstreamChatCompletions(req, upstreamModel)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if req.Stream {
		streamAnthropicViaOpenAI(ctx, w, upstreamBase, upstreamKey, upReq, upstreamModel)
		return
	}

	resp, err := callUpstreamOnce(ctx, upstreamBase, upstreamKey, upReq)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

// ------------------ Anthropic request/response types (minimal) ------------------

type anthropicMessagesRequest struct {
	Model     string          `json:"model"`
	MaxTokens int             `json:"max_tokens"`
	System    any             `json:"system,omitempty"` // string or array of blocks
	Messages  []anthropicMsg  `json:"messages"`
	Tools     []anthropicTool `json:"tools,omitempty"`
	ToolChoice any            `json:"tool_choice,omitempty"`
	Stream    bool            `json:"stream,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
}

type anthropicMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or array of blocks
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// ------------------ OpenAI upstream request types ------------------

type upChatCompletionsRequest struct {
	Model       string         `json:"model"`
	Messages    []upMessage    `json:"messages"`
	Tools       []upTool       `json:"tools,omitempty"`
	ToolChoice  any            `json:"tool_choice,omitempty"`
	MaxTokens   int            `json:"max_tokens,omitempty"`
	Temperature *float64       `json:"temperature,omitempty"`
	Stream      bool           `json:"stream,omitempty"`
}

type upMessage struct {
	Role       string        `json:"role"`
	Content    any           `json:"content,omitempty"`
	ToolCalls  []upToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	Name       string        `json:"name,omitempty"`
}

type upTool struct {
	Type     string        `json:"type"`
	Function upToolFunction `json:"function"`
}

type upToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type upToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"` // "function"
	Function upToolCallFunc  `json:"function"`
}

type upToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ------------------ Conversion ------------------

type anthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

func toUpstreamChatCompletions(req anthropicMessagesRequest, upstreamModel string) (upChatCompletionsRequest, error) {
	var out upChatCompletionsRequest
	out.Model = upstreamModel
	out.Stream = req.Stream
	if req.MaxTokens > 0 {
		out.MaxTokens = req.MaxTokens
	}
	out.Temperature = req.Temperature

	// tools
	if len(req.Tools) > 0 {
		out.Tools = make([]upTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			if t.Name == "" || t.InputSchema == nil {
				return out, fmt.Errorf("invalid tool schema")
			}
			out.Tools = append(out.Tools, upTool{
				Type: "function",
				Function: upToolFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.InputSchema,
				},
			})
		}
	}

	// tool_choice
	if req.ToolChoice != nil {
		// best-effort mapping
		m, _ := req.ToolChoice.(map[string]any)
		if m != nil {
			if typ, _ := m["type"].(string); typ != "" {
				switch typ {
				case "auto":
					out.ToolChoice = "auto"
				case "any":
					out.ToolChoice = "required"
				case "tool":
					name, _ := m["name"].(string)
					if name != "" {
						out.ToolChoice = map[string]any{
							"type": "function",
							"function": map[string]any{"name": name},
						}
					}
				}
			}
		}
	}

	msgs := make([]upMessage, 0, len(req.Messages)+1)

	// system
	if req.System != nil {
		sys := systemToString(req.System)
		if strings.TrimSpace(sys) != "" {
			msgs = append(msgs, upMessage{Role: "system", Content: sys})
		}
	}

	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			userMsgs, toolMsgs, err := anthropicUserContentToUpstream(m.Content)
			if err != nil {
				return out, err
			}
			if userMsgs != "" {
				msgs = append(msgs, upMessage{Role: "user", Content: userMsgs})
			}
			msgs = append(msgs, toolMsgs...)
		case "assistant":
			am, toolCalls, toolResults, err := anthropicAssistantContentToUpstream(m.Content)
			if err != nil {
				return out, err
			}
			if am != "" || len(toolCalls) > 0 {
				msgs = append(msgs, upMessage{Role: "assistant", Content: am, ToolCalls: toolCalls})
			}
			msgs = append(msgs, toolResults...)
		default:
			return out, fmt.Errorf("unsupported role: %s", m.Role)
		}
	}

	out.Messages = msgs
	return out, nil
}

func systemToString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		var sb strings.Builder
		for _, it := range t {
			b, _ := json.Marshal(it)
			var blk anthropicBlock
			if err := json.Unmarshal(b, &blk); err == nil {
				if blk.Type == "text" {
					sb.WriteString(blk.Text)
					sb.WriteString("\n")
				}
			}
		}
		return sb.String()
	default:
		return ""
	}
}

func anthropicUserContentToUpstream(raw json.RawMessage) (userText string, toolMsgs []upMessage, err error) {
	// content can be string or array of blocks
	if len(raw) == 0 {
		return "", nil, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", nil, fmt.Errorf("invalid user content")
		}
		return s, nil, nil
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", nil, fmt.Errorf("invalid user content")
	}
	var sb strings.Builder
	for _, b := range blocks {
		switch b.Type {
		case "text":
			sb.WriteString(b.Text)
		case "tool_result":
			// Convert to OpenAI tool message
			toolText := extractTextFromToolResult(b.Content)
			toolMsgs = append(toolMsgs, upMessage{Role: "tool", ToolCallID: b.ToolUseID, Content: toolText})
		default:
			// ignore unsupported blocks
		}
	}
	return sb.String(), toolMsgs, nil
}

func anthropicAssistantContentToUpstream(raw json.RawMessage) (assistantText string, toolCalls []upToolCall, toolResults []upMessage, err error) {
	if len(raw) == 0 {
		return "", nil, nil, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", nil, nil, fmt.Errorf("invalid assistant content")
		}
		return s, nil, nil, nil
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", nil, nil, fmt.Errorf("invalid assistant content")
	}
	var sb strings.Builder
	for _, b := range blocks {
		switch b.Type {
		case "text":
			sb.WriteString(b.Text)
		case "tool_use":
			id := b.ID
			if id == "" {
				id = "call_" + randHex(12)
			}
			args := "{}"
			if len(b.Input) > 0 {
				args = string(b.Input)
			}
			toolCalls = append(toolCalls, upToolCall{
				ID:   id,
				Type: "function",
				Function: upToolCallFunc{
					Name:      b.Name,
					Arguments: args,
				},
			})
		case "tool_result":
			toolText := extractTextFromToolResult(b.Content)
			toolResults = append(toolResults, upMessage{Role: "tool", ToolCallID: b.ToolUseID, Content: toolText})
		default:
		}
	}
	return sb.String(), toolCalls, toolResults, nil
}

func extractTextFromToolResult(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	if raw[0] == '"' {
		var s string
		_ = json.Unmarshal(raw, &s)
		return s
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var sb strings.Builder
		for _, b := range blocks {
			if b.Type == "text" {
				sb.WriteString(b.Text)
			}
		}
		return sb.String()
	}
	// fallback
	return string(raw)
}

// ------------------ Upstream calling ------------------

type openAIChatCompletionsResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int `json:"index"`
		Message      struct {
			Role      string       `json:"role"`
			Content   string       `json:"content"`
			ToolCalls []upToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

// Anthropic-style response (minimal)
type anthropicMessagesResponse struct {
	ID         string            `json:"id"`
	Type       string            `json:"type"`
	Role       string            `json:"role"`
	Model      string            `json:"model"`
	Content    []map[string]any  `json:"content"`
	StopReason string            `json:"stop_reason,omitempty"`
	Usage      map[string]any    `json:"usage,omitempty"`
}

func callUpstreamOnce(ctx context.Context, upstreamBase, upstreamKey string, upReq upChatCompletionsRequest) (anthropicMessagesResponse, error) {
	var out anthropicMessagesResponse
	upReq.Stream = false

	b, _ := json.Marshal(upReq)
	u := upstreamBase + "/v1/chat/completions"
	hreq, _ := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(b))
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Authorization", "Bearer "+upstreamKey)

	hresp, err := http.DefaultClient.Do(hreq)
	if err != nil {
		return out, err
	}
	defer hresp.Body.Close()
	if hresp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(hresp.Body, 1<<20))
		return out, fmt.Errorf("upstream %d: %s", hresp.StatusCode, strings.TrimSpace(string(body)))
	}

	var oresp openAIChatCompletionsResponse
	if err := json.NewDecoder(hresp.Body).Decode(&oresp); err != nil {
		return out, err
	}
	if len(oresp.Choices) == 0 {
		return out, errors.New("upstream returned no choices")
	}

	choice := oresp.Choices[0]
	msg := choice.Message

	content := make([]map[string]any, 0, 2)
	if strings.TrimSpace(msg.Content) != "" {
		content = append(content, map[string]any{"type": "text", "text": msg.Content})
	}
	if len(msg.ToolCalls) > 0 {
		for _, tc := range msg.ToolCalls {
			content = append(content, map[string]any{
				"type": "tool_use",
				"id":   tc.ID,
				"name": tc.Function.Name,
				"input": mustParseJSON(tc.Function.Arguments),
			})
		}
	}

	stop := mapFinishReason(choice.FinishReason)

	out = anthropicMessagesResponse{
		ID:         "msg_" + randHex(12),
		Type:       "message",
		Role:       "assistant",
		Model:      oresp.Model,
		Content:    content,
		StopReason: stop,
	}
	if oresp.Usage != nil {
		out.Usage = map[string]any{
			"input_tokens":  oresp.Usage.PromptTokens,
			"output_tokens": oresp.Usage.CompletionTokens,
		}
	}
	return out, nil
}

func mustParseJSON(s string) any {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return map[string]any{}
	}
	return v
}

func mapFinishReason(fr string) string {
	switch fr {
	case "tool_calls":
		return "tool_use"
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	default:
		if fr == "" {
			return "end_turn"
		}
		return fr
	}
}

// ------------------ Streaming ------------------

type openAIStreamChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role      string `json:"role,omitempty"`
			Content   string `json:"content,omitempty"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id,omitempty"`
				Type     string `json:"type,omitempty"`
				Function struct {
					Name      string `json:"name,omitempty"`
					Arguments string `json:"arguments,omitempty"`
				} `json:"function,omitempty"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

func streamAnthropicViaOpenAI(ctx context.Context, w http.ResponseWriter, upstreamBase, upstreamKey string, upReq upChatCompletionsRequest, upstreamModel string) {
	upReq.Stream = true
	b, _ := json.Marshal(upReq)
	u := upstreamBase + "/v1/chat/completions"
	hreq, _ := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(b))
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Authorization", "Bearer "+upstreamKey)

	hresp, err := http.DefaultClient.Do(hreq)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	defer hresp.Body.Close()
	if hresp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(hresp.Body, 1<<20))
		writeAnthropicError(w, http.StatusBadGateway, "api_error", fmt.Sprintf("upstream %d: %s", hresp.StatusCode, strings.TrimSpace(string(body))))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	fl, _ := w.(http.Flusher)

	msgID := "msg_" + randHex(12)
	model := upstreamModel

	// Streaming state
	textBlockStarted := false
	toolBlockStarted := map[int]bool{}
	toolCallMeta := map[int]struct{ id, name string }{}
	stopReason := "end_turn"

	sseWrite(w, fl, map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": msgID,
			"type": "message",
			"role": "assistant",
			"model": model,
			"content": []any{},
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	})

	scanner := bufio.NewScanner(hresp.Body)
	// Allow larger chunks
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		c := chunk.Choices[0]

		// Content deltas
		if c.Delta.Content != "" {
			if !textBlockStarted {
				textBlockStarted = true
				sseWrite(w, fl, map[string]any{
					"type": "content_block_start",
					"index": 0,
					"content_block": map[string]any{"type": "text", "text": ""},
				})
			}
			sseWrite(w, fl, map[string]any{
				"type": "content_block_delta",
				"index": 0,
				"delta": map[string]any{"type": "text_delta", "text": c.Delta.Content},
			})
		}

		// Tool call deltas
		if len(c.Delta.ToolCalls) > 0 {
			for _, tc := range c.Delta.ToolCalls {
				idx := tc.Index
				meta := toolCallMeta[idx]
				if tc.ID != "" {
					meta.id = tc.ID
				}
				if tc.Function.Name != "" {
					meta.name = tc.Function.Name
				}
				toolCallMeta[idx] = meta

				anthIdx := 1 + idx
				if !toolBlockStarted[idx] {
					toolBlockStarted[idx] = true
					callID := meta.id
					if callID == "" {
						callID = "call_" + randHex(12)
						meta.id = callID
						toolCallMeta[idx] = meta
					}
					name := meta.name
					if name == "" {
						name = "tool"
					}
					sseWrite(w, fl, map[string]any{
						"type": "content_block_start",
						"index": anthIdx,
						"content_block": map[string]any{
							"type": "tool_use",
							"id":   callID,
							"name": name,
							"input": map[string]any{},
						},
					})
				}
				if tc.Function.Arguments != "" {
					stopReason = "tool_use"
					sseWrite(w, fl, map[string]any{
						"type": "content_block_delta",
						"index": anthIdx,
						"delta": map[string]any{
							"type": "input_json_delta",
							"partial_json": tc.Function.Arguments,
						},
					})
				}
			}
		}

		if c.FinishReason != nil {
			stopReason = mapFinishReason(*c.FinishReason)
			break
		}
	}

	// close blocks
	if textBlockStarted {
		sseWrite(w, fl, map[string]any{"type": "content_block_stop", "index": 0})
	}
	for idx := range toolBlockStarted {
		anthIdx := 1 + idx
		sseWrite(w, fl, map[string]any{"type": "content_block_stop", "index": anthIdx})
	}

	sseWrite(w, fl, map[string]any{
		"type": "message_delta",
		"delta": map[string]any{"stop_reason": stopReason},
		"usage": map[string]any{"output_tokens": 0},
	})
	sseWrite(w, fl, map[string]any{"type": "message_stop"})
}

func sseWrite(w http.ResponseWriter, fl http.Flusher, obj any) {
	b, _ := json.Marshal(obj)
	_, _ = w.Write([]byte("data: "))
	_, _ = w.Write(b)
	_, _ = w.Write([]byte("\n\n"))
	if fl != nil {
		fl.Flush()
	}
}

// ------------------ Helpers ------------------

func writeAnthropicError(w http.ResponseWriter, status int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    typ,
			"message": msg,
		},
	})
}

func getenv(k, def string) string {
	v := strings.TrimSpace(os.Getenv(k))
	if v == "" {
		return def
	}
	return v
}

func mustGetenv(k string) string {
	v := strings.TrimSpace(os.Getenv(k))
	if v == "" {
		log.Fatalf("missing env: %s", k)
	}
	return v
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Truncate(time.Millisecond))
	})
}
