# OpenRouter Bridge

OpenAI-compatible proxy server that routes LLM requests through OpenRouter for low-cost inference. Built for Nomad Internet's Lyzr AI agent stack.

## What It Does

- Exposes `/v1/chat/completions` — drop-in replacement for OpenAI API
- Exposes `/v1/models` — lists all available OpenRouter models
- **Model aliasing** — automatically maps expensive model names (gpt-4o, claude-3-opus) to cheap equivalents (Mistral Small, Gemini Flash)
- Supports streaming responses
- CORS enabled for browser-based integrations

## Model Alias Map

| You Request | Actually Runs | Cost/1M |
|---|---|---|
| `gpt-4o` | `google/gemini-2.0-flash-001` | $0.10 |
| `gpt-4o-mini` | `mistralai/mistral-small-3.1-24b-instruct` | $0.03 |
| `claude-3-5-sonnet-latest` | `google/gemini-2.0-flash-001` | $0.10 |
| `claude-3-opus-20240229` | `qwen/qwq-32b` | $0.15 |
| `gemini-2.5-flash` | `google/gemini-2.5-flash-preview:thinking` | $0.30 |

## Deploy to Railway

1. Push this repo to GitHub
2. Create a new Railway project from the repo
3. Set environment variable: `OPENROUTER_API_KEY=sk-or-v1-...`
4. Deploy — Railway auto-detects Python and uses the Procfile

## Usage

```bash
# Test the bridge
curl https://your-bridge.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Use with Any OpenAI-Compatible Tool

Set base URL to: `https://your-bridge.railway.app/v1`
No API key required (the bridge holds the OpenRouter key server-side).
