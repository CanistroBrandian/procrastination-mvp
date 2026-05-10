from __future__ import annotations

from typing import Any

import httpx

from app.core.errors import TelegramAPIError


class TelegramClient:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.file_url = f"https://api.telegram.org/file/bot{bot_token}"

    async def _post(self, method: str, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(f"{self.base_url}/{method}", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise TelegramAPIError(
                user_message=(
                    f"Telegram Bot API: HTTP {code}. "
                    "Проверьте TELEGRAM_BOT_TOKEN и что бот не удалён."
                ),
                status_code=code,
            ) from exc
        except httpx.TimeoutException as exc:
            raise TelegramAPIError(
                user_message="Telegram: таймаут при запросе к api.telegram.org.",
            ) from exc
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                user_message=(
                    "Telegram: нет связи с api.telegram.org (интернет, VPN, блокировки). "
                    "Проверьте сеть и TELEGRAM_BOT_TOKEN."
                ),
            ) from exc

    async def send_message(self, chat_id: int, text: str) -> None:
        await self._post("sendMessage", {"chat_id": chat_id, "text": text})

    async def set_webhook(self, url: str, secret_token: str) -> None:
        payload = {"url": url, "secret_token": secret_token, "drop_pending_updates": False}
        await self._post("setWebhook", payload)

    async def delete_webhook(self, drop_pending_updates: bool = False) -> None:
        await self._post("deleteWebhook", {"drop_pending_updates": drop_pending_updates})

    async def get_updates(self, offset: int | None = None, timeout: int = 30) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        try:
            async with httpx.AsyncClient(timeout=float(timeout) + 15.0) as client:
                response = await client.get(f"{self.base_url}/getUpdates", params=params)
                response.raise_for_status()
                data = response.json()
            return list(data.get("result") or [])
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 409:
                raise TelegramAPIError(
                    user_message=(
                        "Telegram getUpdates: HTTP 409 — уже установлен webhook или другой процесс получает "
                        "обновления (long polling). Остановите все лишние экземпляры сервера с этим ботом; "
                        "при webhook переключитесь на один режим (polling или webhook)."
                    ),
                    status_code=409,
                ) from exc
            raise TelegramAPIError(
                user_message=(
                    f"Telegram getUpdates: HTTP {code}. Проверьте TELEGRAM_BOT_TOKEN."
                ),
                status_code=code,
            ) from exc
        except httpx.TimeoutException as exc:
            raise TelegramAPIError(user_message="Telegram getUpdates: таймаут.") from exc
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                user_message="Telegram getUpdates: нет соединения с api.telegram.org.",
            ) from exc

    async def get_file_path(self, file_id: str) -> str:
        data = await self._post("getFile", {"file_id": file_id})
        return data["result"]["file_path"]

    def build_file_download_url(self, file_path: str) -> str:
        return f"{self.file_url}/{file_path}"
