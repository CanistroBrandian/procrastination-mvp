from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.services.analytics import AnalyticsService


@dataclass
class FakeProfile:
    telegram_user_id: int = 10


class FakeEventsRepo:
    def __init__(self, events: list[tuple[str, datetime]]) -> None:
        self.events = events

    async def count_between(self, *, telegram_user_id: int, event_type: str, start: datetime, end: datetime):  # noqa: ARG002
        return sum(1 for et, ts in self.events if et == event_type and start <= ts < end)


def _run(coro):
    return asyncio.run(coro)


def test_analytics_snapshot_counts_and_trend():
    now = datetime(2026, 5, 13, 12, 0, 0)
    events = [
        ("created", now - timedelta(hours=2)),
        ("created", now - timedelta(days=1, hours=1)),
        ("created", now - timedelta(days=3)),
        ("completed", now - timedelta(hours=3)),
        ("completed", now - timedelta(days=2)),
        ("completed", now - timedelta(days=9)),
    ]
    svc = AnalyticsService(FakeEventsRepo(events))
    snap = _run(svc.build_snapshot(FakeProfile(), now_utc=now))

    assert snap.created_24h == 1
    assert snap.completed_24h == 1
    assert snap.created_7d == 3
    assert snap.completed_7d == 2
    assert snap.completion_rate_7d == 66.7
    assert snap.completed_trend_vs_prev_7d == 1

