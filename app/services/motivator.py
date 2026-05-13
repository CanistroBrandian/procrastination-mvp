from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.db.models import Persona, UserProfile
from app.db.repositories import IntentHistoryRepository, TaskEventRepository


@dataclass(frozen=True)
class OverdueCard:
    id: str
    name: str
    due: datetime


def _persona_phrase(persona: Persona) -> str:
    if persona == Persona.ELON:
        return "Фокус на шаге сейчас: выбери одну задачу и начни за 5 минут."
    if persona == Persona.ZEN:
        return "Сделай один маленький шаг без перегруза."
    return "Я рядом, давай закроем первый маленький шаг."


def _bucket_hint(overdue_days: int) -> str:
    if overdue_days <= 2:
        return "Просрочка небольшая: хватит 10-15 минут, чтобы сдвинуть задачу."
    if overdue_days <= 7:
        return "Просрочка растёт: начни с одного конкретного результата на сегодня."
    return "Давняя просрочка: упрости задачу и закрой минимально полезный кусок сегодня."


class MotivatorService:
    def __init__(self, events: TaskEventRepository, history: IntentHistoryRepository):
        self.events = events
        self.history = history

    @staticmethod
    def now_in_profile_tz(profile: UserProfile) -> datetime:
        tz_name = (profile.timezone or "").strip() or "Europe/Moscow"
        try:
            tz = ZoneInfo(tz_name)
        except Exception:  # noqa: BLE001
            tz = ZoneInfo("UTC")
        return datetime.now(tz=tz)

    async def can_ping_now(
        self,
        *,
        profile: UserProfile,
        now_utc: datetime,
        max_per_day: int = 3,
        cooldown_minutes: int = 120,
        min_idle_minutes: int = 45,
    ) -> bool:
        sent_today = await self.events.count_today_by_type(
            telegram_user_id=profile.telegram_user_id,
            event_type="motivator_ping",
            now=now_utc,
        )
        if sent_today >= max_per_day:
            return False
        last_ping = await self.events.last_event_at(
            telegram_user_id=profile.telegram_user_id,
            event_type="motivator_ping",
        )
        if last_ping and (now_utc - last_ping) < timedelta(minutes=cooldown_minutes):
            return False
        last_user_activity = await self.history.last_activity_at(profile.telegram_user_id)
        if last_user_activity and (now_utc - last_user_activity) < timedelta(minutes=min_idle_minutes):
            return False
        return True

    @staticmethod
    def pick_top_overdue(cards: list[OverdueCard], now_local: datetime) -> OverdueCard | None:
        if not cards:
            return None
        return sorted(cards, key=lambda c: now_local - c.due, reverse=True)[0]

    @staticmethod
    def render_message(*, profile: UserProfile, card: OverdueCard, now_local: datetime) -> str:
        overdue_days = max(1, (now_local.date() - card.due.date()).days)
        return (
            f"Задача «{card.name}» просрочена на {overdue_days} дн.\n"
            f"{_bucket_hint(overdue_days)}\n"
            f"{_persona_phrase(profile.persona)}"
        )
