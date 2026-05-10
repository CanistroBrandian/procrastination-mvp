from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Persona(str, Enum):
    ELON = "elon"
    ZEN = "zen"
    MOM = "mom"


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    trello_token: Mapped[str | None] = mapped_column(String(256), nullable=True)
    trello_board_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trello_inbox_list_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trello_doing_list_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trello_done_list_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    persona: Mapped[Persona] = mapped_column(SqlEnum(Persona), default=Persona.MOM)
    reminder_cron: Mapped[str] = mapped_column(String(64), default="0 10 * * *")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PendingClarification(Base):
    __tablename__ = "pending_clarifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, index=True)
    draft_action_json: Mapped[str] = mapped_column(Text)
    question: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
