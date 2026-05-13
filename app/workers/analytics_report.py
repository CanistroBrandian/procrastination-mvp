from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.db.models import UserProfile
from app.db.repositories import TaskEventRepository, UserProfileRepository
from app.integrations.telegram import TelegramClient
from app.services.analytics import AnalyticsService, now_in_profile_tz
from app.services.cron_match import cron_matches_now


async def run_daily_analytics_report(
    profile_repo: UserProfileRepository,
    events_repo: TaskEventRepository,
    tg: TelegramClient,
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
        if not cron_matches_now(profile.analytics_cron or "", now_local):
            continue
        snapshot = await analytics.build_snapshot(profile, now_utc=now_utc)
        text = analytics.render_report(snapshot)
        await tg.send_message(profile.telegram_user_id, text)
