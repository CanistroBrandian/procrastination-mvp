"""Пересчитать trello_inbox/doing/done для всех профилей с привязанной доской.

Эквивалент команды в Telegram /link <board_id>, но без чата: заново вызывает
OnboardingService.link_default_board с тем же trello_board_id из БД.

Запуск из корня проекта:
  python scripts/relink_trello_boards.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Корень репозитория в sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import UserProfile
from app.db.session import SessionLocal
from app.integrations.trello import TrelloClient
from app.services.onboarding import OnboardingService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    async with SessionLocal() as session:
        result = await session.execute(
            select(UserProfile).where(UserProfile.trello_board_id.isnot(None)),
        )
        profiles = list(result.scalars().all())

        if not profiles:
            logger.info("Нет профилей с trello_board_id — нечего обновлять.")
            return

        for profile in profiles:
            board_id = (profile.trello_board_id or "").strip()
            if not board_id:
                continue
            token = (profile.trello_token or settings.trello_api_token or "").strip()
            key = (settings.trello_api_key or "").strip()
            trello = TrelloClient(key, token)
            onboarding = OnboardingService(trello)
            try:
                await onboarding.link_default_board(profile, board_id)
                session.add(profile)
                await session.commit()
                await session.refresh(profile)
                logger.info(
                    "Обновлён пользователь telegram_user_id=%s board=%s inbox=%s doing=%s done=%s",
                    profile.telegram_user_id,
                    board_id[:12] + "…",
                    (profile.trello_inbox_list_id or "")[:8],
                    (profile.trello_doing_list_id or "")[:8],
                    (profile.trello_done_list_id or "")[:8],
                )
            except Exception as exc:
                logger.exception(
                    "Ошибка для telegram_user_id=%s: %s",
                    profile.telegram_user_id,
                    exc,
                )
                await session.rollback()


if __name__ == "__main__":
    asyncio.run(main())
