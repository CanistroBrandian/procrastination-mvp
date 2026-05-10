from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.errors import TelegramAPIError
from app.core.llm_client import build_async_openai_client
from app.db.session import SessionLocal
from app.integrations.telegram import TelegramClient
from app.services.telegram_pipeline import process_telegram_update

logger = logging.getLogger(__name__)

# Короче таймаут — быстрее реагируем на остановку сервера.
_POLL_TIMEOUT = 15


async def telegram_polling_loop(settings: Settings, stop_event: asyncio.Event, openai_client: AsyncOpenAI) -> None:
    """Long polling getUpdates — не нужен публичный HTTPS."""
    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN пуст — Telegram polling не запущен.")
        return

    tg = TelegramClient(settings.telegram_bot_token)
    try:
        await tg.delete_webhook(drop_pending_updates=False)
    except Exception as exc:
        logger.warning("deleteWebhook: %s", exc)

    offset: int | None = None
    while not stop_event.is_set():
        try:
            updates = await tg.get_updates(offset=offset, timeout=_POLL_TIMEOUT)
        except TelegramAPIError as exc:
            if getattr(exc, "status_code", None) == 409:
                logger.warning(
                    "getUpdates HTTP 409 — повторный deleteWebhook; убедитесь, что не запущено несколько ботов.",
                )
                try:
                    await tg.delete_webhook(drop_pending_updates=False)
                except Exception as e:
                    logger.warning("deleteWebhook после 409: %s", e)
                await asyncio.sleep(2)
                continue
            logger.exception("getUpdates: %s", exc)
            await asyncio.sleep(3)
            continue
        except Exception as exc:
            logger.exception("getUpdates: %s", exc)
            await asyncio.sleep(3)
            continue
        for u in updates:
            uid = u["update_id"]
            offset = uid + 1
            try:
                async with SessionLocal() as db:
                    await process_telegram_update(db, u, settings, openai_client)
            except Exception:
                logger.exception("Ошибка обработки update_id=%s", uid)
