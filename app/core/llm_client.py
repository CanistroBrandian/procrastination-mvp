from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import Settings


def build_async_openai_client(settings: Settings) -> AsyncOpenAI:
    """Prefer OpenRouter (OpenAI-compatible) when key is set; else OpenAI."""
    if settings.openrouter_api_key:
        headers: dict[str, str] = {}
        if settings.openrouter_http_referer:
            headers["HTTP-Referer"] = settings.openrouter_http_referer
        if settings.openrouter_app_title:
            headers["X-OpenRouter-Title"] = settings.openrouter_app_title
        return AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=str(settings.openrouter_base_url).rstrip("/"),
            default_headers=headers or None,
        )
    return AsyncOpenAI(api_key=settings.openai_api_key)
