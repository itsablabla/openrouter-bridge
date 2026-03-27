"""
OpenRouter Bridge — OpenAI-compatible proxy for Lyzr and any other tool.
Exposes /v1/chat/completions and /v1/models that forward to OpenRouter.
"""

import os
import json
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="OpenRouter Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mistralai/mistral-small-3.1-24b-instruct")

# All aliases route to Qwen3.5-Plus
QWEN_TARGET = "qwen/qwen3.5-plus-02-15"

# Model alias map — every known Lyzr model name maps to Qwen3.5-Plus via OpenRouter
MODEL_ALIASES = {
    # OpenAI aliases → Qwen3.5-Plus
    "gpt-4o": QWEN_TARGET,
    "gpt-4o-mini": QWEN_TARGET,
    "gpt-4-turbo": QWEN_TARGET,
    "gpt-4": QWEN_TARGET,
    "gpt-3.5-turbo": QWEN_TARGET,
    "gpt-4o-2024-11-20": QWEN_TARGET,
    "gpt-4o-2024-08-06": QWEN_TARGET,
    "gpt-4o-mini-2024-07-18": QWEN_TARGET,
    # Anthropic aliases → Qwen3.5-Plus
    "claude-3-5-sonnet-latest": QWEN_TARGET,
    "claude-3-5-sonnet-20241022": QWEN_TARGET,
    "claude-3-opus-20240229": QWEN_TARGET,
    "claude-3-haiku-20240307": QWEN_TARGET,
    "claude-opus-4-5": QWEN_TARGET,
    "claude-haiku-4-5": QWEN_TARGET,
    "claude-3-5-haiku-latest": QWEN_TARGET,
    # Google aliases → Qwen3.5-Plus
    "gemini-2.0-flash": QWEN_TARGET,
    "gemini-2.0-flash-exp": QWEN_TARGET,
    "gemini-2.5-flash": QWEN_TARGET,
    "gemini-3-flash": QWEN_TARGET,
    "gemini-1.5-pro": QWEN_TARGET,
    "gemini-1.5-flash": QWEN_TARGET,
}


def resolve_model(model: str) -> str:
    """Resolve model alias to actual OpenRouter model string."""
    return MODEL_ALIASES.get(model, model)


@app.get("/")
async def root():
    return {
        "service": "OpenRouter Bridge",
        "status": "online",
        "version": "1.0.0",
        "default_model": DEFAULT_MODEL,
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """Return a list of available models from OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{OPENROUTER_BASE}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Forwards to OpenRouter, resolving model aliases for cost optimization.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    body = await request.json()

    # Resolve model alias
    original_model = body.get("model", DEFAULT_MODEL)
    resolved_model = resolve_model(original_model)
    body["model"] = resolved_model

    # Forward headers
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nomadinternet.com",
        "X-Title": "Nomad Internet AI Support",
    }

    stream = body.get("stream", False)

    async with httpx.AsyncClient(timeout=120) as client:
        if stream:
            async def stream_response():
                async with client.stream(
                    "POST",
                    f"{OPENROUTER_BASE}/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"X-Resolved-Model": resolved_model, "X-Original-Model": original_model},
            )
        else:
            resp = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                json=body,
                headers=headers,
            )
            result = resp.json()
            # Inject original model name back so callers aren't confused
            if "model" in result:
                result["_resolved_model"] = result["model"]
                result["model"] = original_model
            return JSONResponse(
                content=result,
                headers={"X-Resolved-Model": resolved_model, "X-Original-Model": original_model},
            )


@app.post("/v1/completions")
async def completions(request: Request):
    """Legacy completions endpoint — converts to chat format."""
    body = await request.json()
    prompt = body.get("prompt", "")
    model = resolve_model(body.get("model", DEFAULT_MODEL))

    chat_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 0.7),
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nomadinternet.com",
        "X-Title": "Nomad Internet AI Support",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            json=chat_body,
            headers=headers,
        )
        return resp.json()
