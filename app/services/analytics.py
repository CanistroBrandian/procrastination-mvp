from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.db.models import UserProfile
from app.db.repositories import TaskEventRepository


@dataclass(frozen=True)
class AnalyticsSnapshot:
    created_24h: int
    completed_24h: int
    completion_rate_24h: float
    created_7d: int
    completed_7d: int
    completion_rate_7d: float
    completed_trend_vs_prev_7d: int


def _rate(done: int, created: int) -> float:
    if created <= 0:
        return 0.0
    return round((done / created) * 100.0, 1)


class AnalyticsService:
    def __init__(self, events: TaskEventRepository):
        self.events = events

    async def build_snapshot(self, profile: UserProfile, now_utc: datetime | None = None) -> AnalyticsSnapshot:
        now = now_utc or datetime.utcnow()
        created_24h = await self.events.count_between(
            telegram_user_id=profile.telegram_user_id,
            event_type="created",
            start=now - timedelta(hours=24),
            end=now,
        )
        completed_24h = await self.events.count_between(
            telegram_user_id=profile.telegram_user_id,
            event_type="completed",
            start=now - timedelta(hours=24),
            end=now,
        )
        created_7d = await self.events.count_between(
            telegram_user_id=profile.telegram_user_id,
            event_type="created",
            start=now - timedelta(days=7),
            end=now,
        )
        completed_7d = await self.events.count_between(
            telegram_user_id=profile.telegram_user_id,
            event_type="completed",
            start=now - timedelta(days=7),
            end=now,
        )
        prev_completed_7d = await self.events.count_between(
            telegram_user_id=profile.telegram_user_id,
            event_type="completed",
            start=now - timedelta(days=14),
            end=now - timedelta(days=7),
        )
        return AnalyticsSnapshot(
            created_24h=created_24h,
            completed_24h=completed_24h,
            completion_rate_24h=_rate(completed_24h, created_24h),
            created_7d=created_7d,
            completed_7d=completed_7d,
            completion_rate_7d=_rate(completed_7d, created_7d),
            completed_trend_vs_prev_7d=completed_7d - prev_completed_7d,
        )

    @staticmethod
    def render_report(snapshot: AnalyticsSnapshot) -> str:
        trend = snapshot.completed_trend_vs_prev_7d
        trend_mark = "↑" if trend > 0 else ("↓" if trend < 0 else "→")
        return (
            "Ежедневная аналитика:\n"
            f"24ч: создано {snapshot.created_24h}, завершено {snapshot.completed_24h}, "
            f"выполнение {snapshot.completion_rate_24h}%\n"
            f"7д: создано {snapshot.created_7d}, завершено {snapshot.completed_7d}, "
            f"выполнение {snapshot.completion_rate_7d}%\n"
            f"Тренд завершений к прошлым 7д: {trend_mark} {trend:+d}"
        )


def now_in_profile_tz(profile: UserProfile) -> datetime:
    tz_name = (profile.timezone or "").strip() or "Europe/Moscow"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    return datetime.now(tz=tz)
