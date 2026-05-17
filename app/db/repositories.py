import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import delete, desc, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ConversationState,
    IntentHistory,
    PendingClarification,
    RoutineTemplate,
    TaskEvent,
    UserProfile,
)


class UserProfileRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, telegram_user_id: int) -> UserProfile:
        profile = await self.get_by_telegram_id(telegram_user_id)
        if profile:
            return profile
        profile = UserProfile(telegram_user_id=telegram_user_id)
        self.session.add(profile)
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def get_by_telegram_id(self, telegram_user_id: int) -> UserProfile | None:
        result = await self.session.execute(
            select(UserProfile).where(UserProfile.telegram_user_id == telegram_user_id)
        )
        return result.scalar_one_or_none()

    async def save(self, profile: UserProfile) -> UserProfile:
        self.session.add(profile)
        await self.session.commit()
        await self.session.refresh(profile)
        return profile


class ClarificationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str) -> None:
        await self.clear(telegram_user_id)
        self.session.add(
            PendingClarification(
                telegram_user_id=telegram_user_id,
                question=question,
                draft_action_json=draft_action_json,
            )
        )
        await self.session.commit()

    async def get(self, telegram_user_id: int) -> PendingClarification | None:
        result = await self.session.execute(
            select(PendingClarification).where(PendingClarification.telegram_user_id == telegram_user_id)
        )
        return result.scalar_one_or_none()

    async def clear(self, telegram_user_id: int) -> None:
        await self.session.execute(
            delete(PendingClarification).where(PendingClarification.telegram_user_id == telegram_user_id)
        )
        await self.session.commit()


class IntentHistoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def append(
        self,
        *,
        telegram_user_id: int,
        user_text: str,
        action_type: str,
        response_text: str,
        action_payload_json: str | None = None,
        resolved_card_id: str | None = None,
        resolved_card_name: str | None = None,
        flow_id: str | None = None,
        resolution_confidence: str | None = None,
    ) -> None:
        self.session.add(
            IntentHistory(
                telegram_user_id=telegram_user_id,
                user_text=user_text,
                action_type=action_type,
                response_text=response_text,
                action_payload_json=action_payload_json,
                resolved_card_id=resolved_card_id,
                resolved_card_name=resolved_card_name,
                flow_id=flow_id,
                resolution_confidence=resolution_confidence,
            ),
        )
        await self.session.commit()

    async def recent_context(self, telegram_user_id: int, *, limit: int = 8) -> str | None:
        result = await self.session.execute(
            select(IntentHistory)
            .where(IntentHistory.telegram_user_id == telegram_user_id)
            .order_by(desc(IntentHistory.created_at))
            .limit(max(1, min(limit, 20)))
        )
        rows = result.scalars().all()
        if not rows:
            return None

        lines: list[str] = []
        for row in reversed(rows):
            if row.action_type in {"none"}:
                continue
            payload = self._brief_payload(row.action_payload_json)
            user_text = (row.user_text or "").strip().replace("\n", " ")
            if len(user_text) > 140:
                user_text = user_text[:137] + "..."
            line = f"- intent={row.action_type}; user='{user_text}'"
            if payload:
                line += f"; fields={payload}"
            lines.append(line)

        if not lines:
            return None
        return "Recent user intent history:\n" + "\n".join(lines[-8:])

    async def last_activity_at(self, telegram_user_id: int) -> datetime | None:
        result = await self.session.execute(
            select(IntentHistory.created_at)
            .where(IntentHistory.telegram_user_id == telegram_user_id)
            .order_by(desc(IntentHistory.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _brief_payload(raw: str | None) -> str:
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            return ""
        if not isinstance(obj, dict):
            return ""
        keys = (
            "card_name",
            "match_text",
            "list_name",
            "due",
            "checklist_item",
            "question",
        )
        chunks: list[str] = []
        for k in keys:
            v = obj.get(k)
            if v is None:
                continue
            s = str(v).strip().replace("\n", " ")
            if len(s) > 60:
                s = s[:57] + "..."
            if s:
                chunks.append(f"{k}={s}")
        return ", ".join(chunks)


_UNSET = object()


class ConversationStateRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, telegram_user_id: int) -> ConversationState | None:
        result = await self.session.execute(
            select(ConversationState).where(ConversationState.telegram_user_id == telegram_user_id),
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        telegram_user_id: int,
        *,
        active_flow: str | None | object = _UNSET,
        active_card_id: str | None | object = _UNSET,
        active_card_name: str | None | object = _UNSET,
        pending_action_json: str | None | object = _UNSET,
        missing_slots_json: str | None | object = _UNSET,
        candidate_cards_json: str | None | object = _UNSET,
        candidate_items_json: str | None | object = _UNSET,
        flow_id: str | None | object = _UNSET,
    ) -> ConversationState:
        state = await self.get(telegram_user_id)
        if state is None:
            state = ConversationState(telegram_user_id=telegram_user_id)
            self.session.add(state)

        if active_flow is not _UNSET:
            state.active_flow = active_flow  # type: ignore[assignment]
        if active_card_id is not _UNSET:
            state.active_card_id = active_card_id  # type: ignore[assignment]
        if active_card_name is not _UNSET:
            state.active_card_name = active_card_name  # type: ignore[assignment]
        if pending_action_json is not _UNSET:
            state.pending_action_json = pending_action_json  # type: ignore[assignment]
        if missing_slots_json is not _UNSET:
            state.missing_slots_json = missing_slots_json  # type: ignore[assignment]
        if candidate_cards_json is not _UNSET:
            state.candidate_cards_json = candidate_cards_json  # type: ignore[assignment]
        if candidate_items_json is not _UNSET:
            state.candidate_items_json = candidate_items_json  # type: ignore[assignment]
        if flow_id is not _UNSET:
            state.flow_id = flow_id  # type: ignore[assignment]

        await self.session.commit()
        await self.session.refresh(state)
        return state


class RoutineTemplateRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_for_user(self, telegram_user_id: int) -> list[RoutineTemplate]:
        result = await self.session.execute(
            select(RoutineTemplate)
            .where(RoutineTemplate.telegram_user_id == telegram_user_id)
            .order_by(desc(RoutineTemplate.created_at)),
        )
        return list(result.scalars().all())

    async def list_active(self) -> list[RoutineTemplate]:
        result = await self.session.execute(
            select(RoutineTemplate).where(RoutineTemplate.is_active == 1),
        )
        return list(result.scalars().all())

    async def get_by_name(self, telegram_user_id: int, name: str) -> RoutineTemplate | None:
        result = await self.session.execute(
            select(RoutineTemplate)
            .where(RoutineTemplate.telegram_user_id == telegram_user_id)
            .where(RoutineTemplate.name == name)
            .limit(1),
        )
        return result.scalar_one_or_none()

    async def create_or_update(
        self,
        *,
        telegram_user_id: int,
        name: str,
        category_key: str | None,
        duration_min: int,
        checklist_template_json: str | None,
        schedule_cron: str,
    ) -> RoutineTemplate:
        row = await self.get_by_name(telegram_user_id, name)
        if row is None:
            row = RoutineTemplate(
                telegram_user_id=telegram_user_id,
                name=name,
                category_key=category_key,
                duration_min=duration_min,
                checklist_template_json=checklist_template_json,
                schedule_cron=schedule_cron,
                is_active=1,
            )
            self.session.add(row)
        else:
            row.category_key = category_key
            row.duration_min = duration_min
            row.checklist_template_json = checklist_template_json
            row.schedule_cron = schedule_cron
            row.is_active = 1
        await self.session.commit()
        await self.session.refresh(row)
        return row

    async def set_active(self, telegram_user_id: int, name: str, is_active: bool) -> RoutineTemplate | None:
        row = await self.get_by_name(telegram_user_id, name)
        if row is None:
            return None
        row.is_active = 1 if is_active else 0
        await self.session.commit()
        await self.session.refresh(row)
        return row

    async def mark_generated(self, template_id: int, date_yyyy_mm_dd: str) -> None:
        result = await self.session.execute(
            select(RoutineTemplate).where(RoutineTemplate.id == template_id).limit(1),
        )
        row = result.scalar_one_or_none()
        if row is None:
            return
        row.last_generated_date = date_yyyy_mm_dd
        await self.session.commit()

    async def mark_generated_if_not_date(self, template_id: int, date_yyyy_mm_dd: str) -> bool:
        stmt = (
            update(RoutineTemplate)
            .where(RoutineTemplate.id == template_id)
            .where(
                or_(
                    RoutineTemplate.last_generated_date.is_(None),
                    RoutineTemplate.last_generated_date != date_yyyy_mm_dd,
                )
            )
            .values(last_generated_date=date_yyyy_mm_dd)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return bool(getattr(result, "rowcount", 0))


class TaskEventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def append(
        self,
        *,
        telegram_user_id: int,
        card_id: str | None,
        event_type: str,
        category_key: str | None = None,
    ) -> None:
        self.session.add(
            TaskEvent(
                telegram_user_id=telegram_user_id,
                card_id=card_id,
                event_type=event_type,
                category_key=category_key,
            ),
        )
        await self.session.commit()

    async def count_between(
        self,
        *,
        telegram_user_id: int,
        event_type: str,
        start: datetime,
        end: datetime,
    ) -> int:
        result = await self.session.execute(
            select(TaskEvent.id)
            .where(TaskEvent.telegram_user_id == telegram_user_id)
            .where(TaskEvent.event_type == event_type)
            .where(TaskEvent.created_at >= start)
            .where(TaskEvent.created_at < end),
        )
        return len(result.scalars().all())

    async def count_today_by_type(self, *, telegram_user_id: int, event_type: str, now: datetime) -> int:
        day_start = datetime(now.year, now.month, now.day)
        day_end = day_start + timedelta(days=1)
        return await self.count_between(
            telegram_user_id=telegram_user_id,
            event_type=event_type,
            start=day_start,
            end=day_end,
        )

    async def count_local_day_by_type(
        self,
        *,
        telegram_user_id: int,
        event_type: str,
        now_utc: datetime,
        tz_name: str,
    ) -> int:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")
        aware_utc = now_utc.replace(tzinfo=ZoneInfo("UTC")) if now_utc.tzinfo is None else now_utc.astimezone(ZoneInfo("UTC"))
        local_now = aware_utc.astimezone(tz)
        local_start = datetime(local_now.year, local_now.month, local_now.day, tzinfo=tz)
        local_end = local_start + timedelta(days=1)
        start_utc = local_start.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = local_end.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        return await self.count_between(
            telegram_user_id=telegram_user_id,
            event_type=event_type,
            start=start_utc,
            end=end_utc,
        )

    async def last_event_at(self, *, telegram_user_id: int, event_type: str) -> datetime | None:
        result = await self.session.execute(
            select(TaskEvent.created_at)
            .where(TaskEvent.telegram_user_id == telegram_user_id)
            .where(TaskEvent.event_type == event_type)
            .order_by(desc(TaskEvent.created_at))
            .limit(1),
        )
        return result.scalar_one_or_none()
