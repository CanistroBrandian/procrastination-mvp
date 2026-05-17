from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select

from app.db.models import UserProfile
from app.db.repositories import TaskEventRepository, UserProfileRepository
from app.integrations.telegram import TelegramClient
from app.services.analytics import AnalyticsService, now_in_profile_tz
from app.services.cron_match import cron_matches_window, normalize_cron_or_default

logger = logging.getLogger(__name__)


async def run_daily_analytics_report(
    profile_repo: UserProfileRepository,
    events_repo: TaskEventRepository,
    tg: TelegramClient,
    *,
    tick_minutes: int = 10,
    default_analytics_cron: str = "30 21 * * *",
) -> None:
    session = profile_repo.session
    users_result = await session.execute(select(UserProfile))
    users = users_result.scalars().all()
    analytics = AnalyticsService(events_repo)
    now_utc = datetime.utcnow()
    for profile in users:
        if not profile.trello_board_id:
            continue
        now_local = now_in_profile_tz(profile)
        effective_cron, fallback_used = normalize_cron_or_default(profile.analytics_cron, default_analytics_cron)
        if fallback_used:
            logger.warning(
                "invalid profile analytics_cron for user=%s: %r, fallback=%r",
                profile.telegram_user_id,
                profile.analytics_cron,
                effective_cron,
            )
        if not cron_matches_window(effective_cron, now_local, window_minutes=tick_minutes):
            continue
        sent_today = await events_repo.count_local_day_by_type(
            telegram_user_id=profile.telegram_user_id,
            event_type="analytics_report_sent",
            now_utc=now_utc,
            tz_name=(profile.timezone or "Europe/Moscow"),
        )
        if sent_today > 0:
            continue
        snapshot = await analytics.build_snapshot(profile, now_utc=now_utc)
        text = analytics.render_report(snapshot)
        await tg.send_message(profile.telegram_user_id, text)
        await events_repo.append(
            telegram_user_id=profile.telegram_user_id,
            card_id=None,
            event_type="analytics_report_sent",
            category_key=None,
        )
