# ChatJimmy API Completions Proxy

![ChatJimmy](https://chatjimmy.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fchat_jimmy.7c9fc307.png&w=640&q=75)

An OpenAI-compatible proxy that lets [OpenCode](https://opencode.ai) (and any OpenAI-compatible client) use [ChatJimmy](https://chatjimmy.ai)'s hardware-accelerated Llama 3.1 8B, powered by [Taalas](https://taalas.com)' custom silicon running at ~17K tokens/sec.

## Usage

```bash
# Start the proxy
python proxy.py

# Or with options
python proxy.py --port 4100 --log-file proxy.log
```

Then point OpenCode (or any OpenAI-compatible client) at `http://localhost:4100/v1` using the config in `opencode.json`.

## API

The proxy exposes standard OpenAI-compatible endpoints:

### `GET /v1/models`

Returns available models.

### `POST /v1/chat/completions`

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `llama3.1-8B` | Model ID |
| `messages` | array | required | Array of `{role, content}` messages (`system`, `user`, `assistant`, `tool`) |
| `stream` | boolean | `false` | Enable SSE streaming |
| `tools` | array | `[]` | OpenAI-format tool/function definitions |
| `tool_choice` | string \| object | `"auto"` | `"auto"`, `"none"`, `"required"`, or `{"type": "function", "function": {"name": "..."}}` |

**Response format** (non-streaming):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "llama3.1-8B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "...",
      "tool_calls": [{"id": "call_...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
    },
    "finish_reason": "stop" | "tool_calls"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

When `stream: true`, the proxy returns SSE chunks in the standard `chat.completion.chunk` format.

## Limitations

- **Model**: Only Llama 3.1 8B is available (aggressively quantized 3-bit/6-bit, so quality is below GPU baselines)
- **System prompt size**: ChatJimmy silently returns empty responses when the system prompt exceeds ~30K characters; the proxy truncates at 28K as a safeguard
- **Tool calling**: Emulated via `<tool_call>` XML tags in the prompt rather than native function calling, since the underlying model doesn't support it natively
- **No authentication**: ChatJimmy's API is currently open beta with no API key required
- **No streaming from upstream**: The proxy buffers the full ChatJimmy response before streaming it back to the client, so time-to-first-token equals full generation time
