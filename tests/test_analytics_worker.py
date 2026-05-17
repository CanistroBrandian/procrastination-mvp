from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from app.db.models import UserProfile
from app.workers.analytics_report import run_daily_analytics_report


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)


class _Session:
    def __init__(self, users):
        self.users = users

    async def execute(self, stmt):  # noqa: ARG002
        return _Result(self.users)


class _ProfileRepo:
    def __init__(self, users):
        self.session = _Session(users)


class _EventsRepo:
    def __init__(self):
        self.sent_today = 0
        self.created = 3
        self.completed = 2
        self.appended: list[str] = []

    async def count_local_day_by_type(self, **kwargs):
        return self.sent_today

    async def count_between(self, *, event_type: str, **kwargs):  # noqa: ARG002
        return self.created if event_type == "created" else self.completed

    async def append(self, *, event_type: str, **kwargs):  # noqa: ARG002
        self.appended.append(event_type)
        if event_type == "analytics_report_sent":
            self.sent_today += 1


class _TG:
    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, chat_id: int, text: str):  # noqa: ARG002
        self.messages.append(text)


def _run(coro):
    return asyncio.run(coro)


def test_analytics_report_sent_once_per_local_day():
    profile = UserProfile(
        telegram_user_id=22,
        trello_board_id="B1",
        timezone="Europe/Moscow",
        analytics_cron="* * * * *",
    )
    repo = _ProfileRepo([profile])
    events = _EventsRepo()
    tg = _TG()

    _run(run_daily_analytics_report(repo, events, tg, tick_minutes=10))
    _run(run_daily_analytics_report(repo, events, tg, tick_minutes=10))

    assert len(tg.messages) == 1
    assert events.appended.count("analytics_report_sent") == 1


def test_analytics_worker_fallbacks_invalid_profile_cron():
    profile = UserProfile(
        telegram_user_id=23,
        trello_board_id="B2",
        timezone="Europe/Moscow",
        analytics_cron="invalid cron",
    )
    repo = _ProfileRepo([profile])
    events = _EventsRepo()
    tg = _TG()

    _run(
        run_daily_analytics_report(
            repo,
            events,
            tg,
            tick_minutes=10,
            default_analytics_cron="* * * * *",
        )
    )

    assert len(tg.messages) == 1
    assert events.appended.count("analytics_report_sent") == 1
