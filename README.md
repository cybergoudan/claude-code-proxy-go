# Claude Code Proxy (Go)

把 **Claude Code** 需要的 **Anthropic Messages API**（`/v1/messages`）转成你现有的 **OpenAI-compatible Chat Completions API**（`/v1/chat/completions`）。

目标：让 Claude Code 继续使用它的**复杂本地工具链**（bash/git/文件/MCP），同时模型走你自己的 `baseURL + key`。

> 说明：Claude Code 的“复杂工具”大部分是本地执行；本项目重点是把 **tool_use / tool_result** + **SSE 流式**接起来。

## 快速开始

### 1) 启动代理

```bash
export UPSTREAM_BASE_URL="https://YOUR_HOST/v1"
export UPSTREAM_API_KEY="YOUR_KEY"
export UPSTREAM_MODEL="gpt-5.2"   # 你要用的模型名
export LISTEN=":8088"

go run .
```

### 2) 让 Claude Code 走代理

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8088"
export ANTHROPIC_API_KEY="dummy"  # 代理不会用它

# 关闭 Anthropic 特有 tool_reference 自动检索（避免协议不兼容）
export ENABLE_TOOL_SEARCH="standard"

claude
```

## 支持范围（当前版本）

- `POST /v1/messages`
- Anthropic tools → OpenAI `tools:function`
- OpenAI tool_calls → Anthropic `tool_use`
- OpenAI SSE → Anthropic SSE（message_start/content_block_delta/message_stop）

## 限制

- 只转发到 OpenAI-compatible **chat.completions**；如果你的上游是 `/v1/responses`，需要再加一层适配。
- Anthropic 的 `tool_reference` / 动态工具检索不在本项目范围内（建议用 `ENABLE_TOOL_SEARCH=standard`）。

## 安全

- 不会打印 `UPSTREAM_API_KEY`
