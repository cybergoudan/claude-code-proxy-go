# claude-code-proxy-go

A small Go proxy that lets **Claude Code** talk to an **OpenAI-compatible** upstream by exposing an **Anthropic Messages API**-compatible endpoint.

- Exposes: `POST /v1/messages` (Anthropic-style)
- Forwards to: `POST /chat/completions` on your upstream (OpenAI Chat Completions style)
- Supports streaming (SSE) and tool calling (best-effort)

> Claude Code's "complex tools" (bash/git/files/MCP) are mostly executed **locally**; this proxy mainly bridges the **model API protocol**.

## Why
Claude Code natively expects Anthropic's Messages API. If you already have an OpenAI-compatible endpoint (self-hosted gateway, proxy, private vendor), this provides a compatibility layer.

## Install

### Option A: run from source

```bash
git clone https://github.com/cybergoudan/claude-code-proxy-go
cd claude-code-proxy-go

go run .
```

### Option B: build binary

```bash
go build -o claude-code-proxy .
./claude-code-proxy
```

## Configuration (required)

This proxy **does not** hardcode any URL or key. You must pass them via env vars.

```bash
export UPSTREAM_BASE_URL="https://YOUR_HOST/v1"   # upstream base, typically ends with /v1
export UPSTREAM_API_KEY="YOUR_KEY"               # upstream bearer token
export UPSTREAM_MODEL="gpt-5.2"                  # model id used upstream
export LISTEN=":8088"                            # local listen address

./claude-code-proxy
```

Health check:

```bash
curl -fsS http://127.0.0.1:8088/healthz
```

## Use with Claude Code

Tell Claude Code to use the proxy as its Anthropic endpoint:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8088"
export ANTHROPIC_API_KEY="dummy"  # unused by the proxy

# Recommended: disable Anthropic-specific tool search / tool_reference behaviors.
# This avoids protocol mismatches with non-Anthropic upstreams.
export ENABLE_TOOL_SEARCH="standard"

claude
```

## Notes / Limitations

- This is a **best-effort** protocol bridge.
- Upstream must support **OpenAI Chat Completions** + streaming (`stream:true`) and tool calling (`tools` / `tool_calls`).
- Some Anthropic-only features (e.g. `tool_reference` search) are intentionally not implemented.

## Security

- Keep your `UPSTREAM_API_KEY` in environment variables or a secret store.
- Do not commit your keys to git.

## License

MIT
