from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError

from app.core.config import Settings
from app.core.errors import OpenAITranscriptionError, OpenRouterTranscriptionError


def _audio_format_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".ogg", ".oga"}:
        return "ogg"
    if ext == ".mp3":
        return "mp3"
    if ext in {".m4a", ".mp4"}:
        return "m4a"
    if ext == ".wav":
        return "wav"
    return "ogg"


class TranscriptionService(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError


class OpenRouterTranscriptionService(TranscriptionService):
    """OpenRouter `/audio/transcriptions` — base64 JSON (совместимо с их примером)."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._url = f"{str(settings.openrouter_base_url).rstrip('/')}/audio/transcriptions"

    async def transcribe(self, audio_path: Path) -> str:
        data_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
        fmt = _audio_format_for_path(audio_path)
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.openrouter_http_referer:
            headers["HTTP-Referer"] = self._settings.openrouter_http_referer
        if self._settings.openrouter_app_title:
            headers["X-OpenRouter-Title"] = self._settings.openrouter_app_title

        payload = {
            "model": self._settings.whisper_model,
            "input_audio": {"data": data_b64, "format": fmt},
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self._url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            raise OpenRouterTranscriptionError(
                f"Whisper (OpenRouter): HTTP {exc.response.status_code}. Проверьте ключ и WHISPER_MODEL.",
            ) from exc
        except httpx.RequestError as exc:
            raise OpenRouterTranscriptionError(
                "Whisper (OpenRouter): нет соединения. Проверьте интернет и OPENROUTER_BASE_URL.",
            ) from exc
        text = body.get("text") if isinstance(body, dict) else None
        if not text:
            raise OpenRouterTranscriptionError(
                f"Whisper (OpenRouter): неожиданный ответ: {body!r}",
            )
        return str(text).strip()


class OpenAITranscriptionService(TranscriptionService):
    """Прямой OpenAI Whisper API (multipart file)."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def transcribe(self, audio_path: Path) -> str:
        try:
            with audio_path.open("rb") as audio_file:
                transcript = await self._client.audio.transcriptions.create(
                    model=self._model,
                    file=audio_file,
                )
            return transcript.text.strip()
        except AuthenticationError as exc:
            raise OpenAITranscriptionError(
                "Whisper (OpenAI): неверный OPENAI_API_KEY.",
            ) from exc
        except APIConnectionError as exc:
            raise OpenAITranscriptionError(
                "Whisper (OpenAI): нет соединения с API.",
            ) from exc
        except APIStatusError as exc:
            raise OpenAITranscriptionError(
                f"Whisper (OpenAI): ошибка {exc.status_code}. Проверьте WHISPER_MODEL.",
            ) from exc


def build_transcription_service(settings: Settings, openai_client: AsyncOpenAI) -> TranscriptionService:
    if settings.openrouter_api_key:
        return OpenRouterTranscriptionService(settings)
    return OpenAITranscriptionService(openai_client, settings.whisper_model)
