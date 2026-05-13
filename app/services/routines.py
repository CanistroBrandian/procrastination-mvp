from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.db.models import RoutineTemplate, UserProfile
from app.db.repositories import RoutineTemplateRepository
from app.integrations.trello import TrelloClient
from app.services.categories import normalize_category_key
from app.services.cron_match import cron_matches_now


@dataclass(frozen=True)
class RoutineCreatePayload:
    name: str
    category_key: str | None
    duration_min: int
    checklist_items: list[str]
    schedule_cron: str


class RoutineService:
    def __init__(self, repo: RoutineTemplateRepository, trello_client: TrelloClient):
        self.repo = repo
        self.trello_client = trello_client

    async def create_or_update(self, profile: UserProfile, payload: RoutineCreatePayload) -> RoutineTemplate:
        category_key = normalize_category_key(payload.category_key)
        checklist_json = json.dumps(payload.checklist_items, ensure_ascii=False) if payload.checklist_items else None
        return await self.repo.create_or_update(
            telegram_user_id=profile.telegram_user_id,
            name=payload.name.strip(),
            category_key=category_key,
            duration_min=max(1, payload.duration_min),
            checklist_template_json=checklist_json,
            schedule_cron=payload.schedule_cron.strip(),
        )

    async def pause(self, profile: UserProfile, name: str) -> bool:
        row = await self.repo.set_active(profile.telegram_user_id, name.strip(), False)
        return row is not None

    async def resume(self, profile: UserProfile, name: str) -> bool:
        row = await self.repo.set_active(profile.telegram_user_id, name.strip(), True)
        return row is not None

    async def list_for_user(self, profile: UserProfile) -> list[RoutineTemplate]:
        return await self.repo.list_for_user(profile.telegram_user_id)

    @staticmethod
    def should_run_template(template: RoutineTemplate, now_local: datetime) -> bool:
        if int(template.is_active or 0) != 1:
            return False
        if not cron_matches_now(template.schedule_cron or "", now_local):
            return False
        today = now_local.strftime("%Y-%m-%d")
        return (template.last_generated_date or "") != today

    @staticmethod
    def parse_checklist(template: RoutineTemplate) -> list[str]:
        raw = template.checklist_template_json
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(payload, list):
            return []
        return [str(x).strip() for x in payload if str(x).strip()]

    async def generate_card_for_template(
        self,
        *,
        profile: UserProfile,
        template: RoutineTemplate,
        now_local: datetime,
    ) -> dict[str, object] | None:
        if not profile.trello_inbox_list_id:
            return None
        today = now_local.strftime("%Y-%m-%d")
        title = f"{template.name} — {today}"
        desc = f"Рутина на день: {template.duration_min} мин."
        card = await self.trello_client.create_card(profile.trello_inbox_list_id, title, desc=desc)
        card_id = str(card.get("id") or "")
        if card_id:
            checklist = self.parse_checklist(template)
            if checklist:
                cl = await self.trello_client.add_checklist(card_id, "Шаги")
                cl_id = str(cl.get("id") or "")
                if cl_id:
                    for item in checklist:
                        await self.trello_client.add_check_item(cl_id, item)
            await self.repo.mark_generated(template.id, today)
        return card


def now_in_user_tz(profile: UserProfile, *, fallback_tz: str = "Europe/Moscow") -> datetime:
    tz_name = (profile.timezone or "").strip() or fallback_tz
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    return datetime.now(tz=tz)
