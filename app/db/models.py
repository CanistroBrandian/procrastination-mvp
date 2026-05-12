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


class IntentHistory(Base):
    __tablename__ = "intent_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, index=True)
    user_text: Mapped[str] = mapped_column(Text)
    action_type: Mapped[str] = mapped_column(String(64))
    action_payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_card_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_card_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    flow_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolution_confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    response_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ConversationState(Base):
    __tablename__ = "conversation_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    active_flow: Mapped[str | None] = mapped_column(String(64), nullable=True)
    active_card_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    active_card_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    pending_action_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    missing_slots_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_cards_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_items_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    flow_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
