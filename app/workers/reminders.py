from __future__ import annotations

from sqlalchemy import select

from app.db.models import UserProfile
from app.db.repositories import UserProfileRepository
from app.integrations.telegram import TelegramClient
from app.integrations.trello import TrelloClient


async def run_reminders(profile_repo: UserProfileRepository, tg: TelegramClient, trello: TrelloClient) -> None:
    # MVP fallback: ping every user with connected board.
    # Next iteration can inspect list/card due dates and inactivity windows.
    session = profile_repo.session
    _ = trello  # Placeholder for future card-state aware reminder checks.
    result = await session.execute(select(UserProfile))
    users = result.scalars().all()
    for profile in users:
        if not profile.trello_board_id:
            continue
        reminder_text = {
            "elon": "Ты теряешь темп. Выбери одну задачу и начни в ближайшие 5 минут.",
            "zen": "Сделай паузу, выбери самый маленький следующий шаг и выполни его сейчас.",
            "mom": "Я в тебя верю. Один маленький шаг по задаче уже будет победой.",
        }[profile.persona.value]
        await tg.send_message(profile.telegram_user_id, reminder_text)
