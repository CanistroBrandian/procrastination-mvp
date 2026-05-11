import json

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IntentHistory, PendingClarification, UserProfile


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
    ) -> None:
        self.session.add(
            IntentHistory(
                telegram_user_id=telegram_user_id,
                user_text=user_text,
                action_type=action_type,
                response_text=response_text,
                action_payload_json=action_payload_json,
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
