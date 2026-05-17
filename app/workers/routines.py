from __future__ import annotations

import json
import logging

from sqlalchemy import select

from app.db.models import UserProfile
from app.db.repositories import RoutineTemplateRepository, TaskEventRepository, UserProfileRepository
from app.integrations.trello import TrelloClient
from app.services.categories import CategoryService
from app.services.cron_match import normalize_cron_or_default
from app.services.routines import RoutineService, now_in_user_tz

logger = logging.getLogger(__name__)


async def run_routine_generation(
    profile_repo: UserProfileRepository,
    routines_repo: RoutineTemplateRepository,
    task_events: TaskEventRepository,
    trello: TrelloClient,
    *,
    tick_minutes: int = 10,
    default_routine_cron: str = "0 8 * * *",
) -> None:
    session = profile_repo.session
    users_result = await session.execute(select(UserProfile))
    users = users_result.scalars().all()
    by_user = {}
    for tpl in await routines_repo.list_active():
        by_user.setdefault(tpl.telegram_user_id, []).append(tpl)

    for profile in users:
        templates = by_user.get(profile.telegram_user_id, [])
        if not templates:
            continue
        user_trello = TrelloClient(trello.api_key, profile.trello_token or trello.api_token)
        service = RoutineService(routines_repo, user_trello)
        category_service = CategoryService(user_trello)
        now_local = now_in_user_tz(profile, fallback_tz="Europe/Moscow")
        profile_cron, profile_fallback = normalize_cron_or_default(profile.routine_cron, default_routine_cron)
        if profile_fallback:
            logger.warning(
                "invalid profile routine_cron for user=%s: %r, fallback=%r",
                profile.telegram_user_id,
                profile.routine_cron,
                profile_cron,
            )
        for template in templates:
            template_cron, template_fallback = normalize_cron_or_default(template.schedule_cron, profile_cron)
            if template_fallback:
                logger.warning(
                    "invalid template cron for user=%s template=%s: %r, fallback=%r",
                    profile.telegram_user_id,
                    template.name,
                    template.schedule_cron,
                    template_cron,
                )
            if not service.should_run_template(
                template,
                now_local,
                window_minutes=tick_minutes,
                effective_cron=template_cron,
            ):
                continue
            card = await service.generate_card_for_template(
                profile=profile,
                template=template,
                now_local=now_local,
            )
            card_id = str((card or {}).get("id") or "")
            if not card_id:
                continue
            if template.category_key:
                try:
                    await category_service.apply_category(profile, card_id, template.category_key)
                except Exception:  # noqa: BLE001
                    logger.exception("routine category apply failed user=%s template=%s", profile.telegram_user_id, template.name)
            await task_events.append(
                telegram_user_id=profile.telegram_user_id,
                card_id=card_id,
                event_type="created",
                category_key=template.category_key,
            )
            await task_events.append(
                telegram_user_id=profile.telegram_user_id,
                card_id=card_id,
                event_type="routine_generated",
                category_key=template.category_key,
            )
            logger.info(
                "routine generated user=%s template=%s card_id=%s checklist=%s",
                profile.telegram_user_id,
                template.name,
                card_id,
                json.dumps(service.parse_checklist(template), ensure_ascii=False),
            )
