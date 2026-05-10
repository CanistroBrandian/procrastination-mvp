from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PendingClarification, UserProfile


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
