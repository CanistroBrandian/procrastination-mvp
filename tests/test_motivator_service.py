from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.db.models import Persona
from app.services.motivator import MotivatorService


@dataclass
class FakeProfile:
    telegram_user_id: int = 42
    timezone: str = "Europe/Moscow"
    persona: Persona = Persona.MOM


class FakeEventsRepo:
    def __init__(self, sent_today: int = 0, last_ping: datetime | None = None) -> None:
        self.sent_today = sent_today
        self.last_ping = last_ping

    async def count_today_by_type(self, *, telegram_user_id: int, event_type: str, now: datetime):  # noqa: ARG002
        return self.sent_today if event_type == "motivator_ping" else 0

    async def last_event_at(self, *, telegram_user_id: int, event_type: str):  # noqa: ARG002
        if event_type == "motivator_ping":
            return self.last_ping
        return None


class FakeHistoryRepo:
    def __init__(self, last_user_activity: datetime | None = None) -> None:
        self.last_user_activity = last_user_activity

    async def last_activity_at(self, telegram_user_id: int):  # noqa: ARG002
        return self.last_user_activity


def _run(coro):
    return asyncio.run(coro)


def test_motivator_anti_spam_limits():
    now = datetime(2026, 5, 13, 12, 0, 0)
    profile = FakeProfile()

    svc_limit = MotivatorService(FakeEventsRepo(sent_today=3), FakeHistoryRepo())
    assert _run(svc_limit.can_ping_now(profile=profile, now_utc=now)) is False

    svc_cooldown = MotivatorService(FakeEventsRepo(sent_today=1, last_ping=now - timedelta(minutes=30)), FakeHistoryRepo())
    assert _run(svc_cooldown.can_ping_now(profile=profile, now_utc=now)) is False

    svc_recent_activity = MotivatorService(
        FakeEventsRepo(sent_today=0, last_ping=None),
        FakeHistoryRepo(last_user_activity=now - timedelta(minutes=15)),
    )
    assert _run(svc_recent_activity.can_ping_now(profile=profile, now_utc=now)) is False

    svc_ok = MotivatorService(
        FakeEventsRepo(sent_today=1, last_ping=now - timedelta(hours=4)),
        FakeHistoryRepo(last_user_activity=now - timedelta(hours=2)),
    )
    assert _run(svc_ok.can_ping_now(profile=profile, now_utc=now)) is True

