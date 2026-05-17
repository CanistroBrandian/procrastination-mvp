from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging

from sqlalchemy import select

from app.db.models import UserProfile
from app.db.repositories import IntentHistoryRepository, TaskEventRepository, UserProfileRepository
from app.integrations.telegram import TelegramClient
from app.integrations.trello import TrelloClient
from app.services.motivator import MotivatorService, OverdueCard
from app.services.trello_board_filters import drop_cards_in_archived_lists
from app.services.cron_match import cron_matches_window, normalize_cron_or_default

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OverdueCandidate:
    id: str
    name: str
    due: datetime


def _parse_due_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        due = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if due.tzinfo is None:
            return due.replace(tzinfo=UTC)
        return due
    except ValueError:
        return None


async def run_overdue_motivator(
    profile_repo: UserProfileRepository,
    events_repo: TaskEventRepository,
    history_repo: IntentHistoryRepository,
    tg: TelegramClient,
    trello: TrelloClient,
    *,
    tick_minutes: int = 10,
    default_motivator_cron: str = "0 11,16,20 * * *",
) -> None:
    session = profile_repo.session
    users_result = await session.execute(select(UserProfile))
    users = users_result.scalars().all()
    motivator = MotivatorService(events_repo, history_repo)
    now_utc = datetime.utcnow()

    for profile in users:
        if not profile.trello_board_id:
            continue
        now_local = motivator.now_in_profile_tz(profile)
        effective_cron, fallback_used = normalize_cron_or_default(profile.motivator_cron_windows, default_motivator_cron)
        if fallback_used:
            logger.warning(
                "invalid profile motivator_cron_windows for user=%s: %r, fallback=%r",
                profile.telegram_user_id,
                profile.motivator_cron_windows,
                effective_cron,
            )
        if not cron_matches_window(effective_cron, now_local, window_minutes=tick_minutes):
            continue
        if not await motivator.can_ping_now(profile=profile, now_utc=now_utc):
            continue
        user_trello = TrelloClient(trello.api_key, profile.trello_token or trello.api_token)
        lists = await user_trello.list_lists(profile.trello_board_id)
        cards = await user_trello.list_board_cards_summary(profile.trello_board_id)
        cards = drop_cards_in_archived_lists(cards, lists)
        overdue: list[OverdueCard] = []
        for card in cards:
            due = _parse_due_iso(str(card.get("due") or ""))
            if due is None:
                continue
            due_complete = bool(card.get("dueComplete"))
            if due_complete:
                continue
            card_list_id = str(card.get("idList") or "")
            if profile.trello_done_list_id and card_list_id == profile.trello_done_list_id:
                continue
            if due < now_local:
                overdue.append(
                    OverdueCard(
                        id=str(card.get("id") or ""),
                        name=str(card.get("name") or "Без названия"),
                        due=due,
                    ),
                )
        top = motivator.pick_top_overdue(overdue, now_local)
        if top is None:
            continue
        text = motivator.render_message(profile=profile, card=top, now_local=now_local)
        await tg.send_message(profile.telegram_user_id, text)
        await events_repo.append(
            telegram_user_id=profile.telegram_user_id,
            card_id=top.id,
            event_type="motivator_ping",
        )
