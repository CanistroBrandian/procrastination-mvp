from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.db.models import Persona
from app.services import telegram_pipeline as pipeline


@dataclass
class _Settings:
    telegram_bot_token: str = "token"
    chat_model: str = "model"
    trello_api_key: str = "k"
    trello_api_token: str = "t"
    default_trello_board_id: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""


@dataclass
class _Profile:
    telegram_user_id: int = 10
    persona: Persona = Persona.MOM
    trello_board_id: str | None = None
    trello_token: str | None = None
    trello_inbox_list_id: str | None = None
    trello_doing_list_id: str | None = None
    trello_done_list_id: str | None = None


class _FakeTelegramClient:
    sent: list[str] = []

    def __init__(self, bot_token: str):  # noqa: ARG002
        pass

    async def send_message(self, chat_id: int, text: str):  # noqa: ARG002
        self.__class__.sent.append(text)


class _FakeProfileRepo:
    def __init__(self, session):  # noqa: ARG002
        self.profile = _Profile()

    async def get_or_create(self, user_id: int):  # noqa: ARG002
        return self.profile

    async def save(self, profile):  # noqa: ARG002
        return profile


class _FakeClarificationRepo:
    def __init__(self, session):  # noqa: ARG002
        self.cleared: list[int] = []

    async def get(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def clear(self, telegram_user_id: int):
        self.cleared.append(telegram_user_id)

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str):  # noqa: ARG002
        raise AssertionError("attachment clarification should not be used for board URL")


class _FakeClarificationRepoWithPending(_FakeClarificationRepo):
    async def get(self, telegram_user_id: int):  # noqa: ARG002
        return type(
            "Pending",
            (),
            {"question": "pending", "draft_action_json": '{"action_type":"ask_for_clarification"}'},
        )()


class _FakeConversationStateRepo:
    def __init__(self, session):  # noqa: ARG002
        pass

    async def upsert(self, telegram_user_id: int, **kwargs):  # noqa: ARG002
        return None


class _FakeIntentHistoryRepo:
    def __init__(self, session):  # noqa: ARG002
        pass

    async def recent_context(self, telegram_user_id: int, *, limit: int = 8):  # noqa: ARG002
        return None

    async def append(self, **kwargs):  # noqa: ARG002
        return None


class _FakeRoutineRepo:
    def __init__(self, session):  # noqa: ARG002
        pass


class _FakeTaskEventRepo:
    def __init__(self, session):  # noqa: ARG002
        pass


class _FakeTrelloClient:
    def __init__(self, api_key: str, api_token: str):  # noqa: ARG002
        self.api_key = api_key
        self.api_token = api_token


class _FakeOnboardingService:
    called_with: list[str] = []

    def __init__(self, trello_client):  # noqa: ARG002
        pass

    async def link_default_board(self, profile, board_id: str):
        self.__class__.called_with.append(board_id)
        profile.trello_board_id = board_id


class _FakeTaskOrchestrator:
    def __init__(self, **kwargs):  # noqa: ARG002
        pass

    async def process_text(self, profile, text, *, intent_context=None):  # noqa: ARG002
        raise AssertionError("orchestrator should not be called for board URL message")


def _run(coro):
    return asyncio.run(coro)


def _setup(monkeypatch, clar_repo_cls):
    monkeypatch.setattr(pipeline, "TelegramClient", _FakeTelegramClient)
    monkeypatch.setattr(pipeline, "UserProfileRepository", _FakeProfileRepo)
    monkeypatch.setattr(pipeline, "ClarificationRepository", clar_repo_cls)
    monkeypatch.setattr(pipeline, "ConversationStateRepository", _FakeConversationStateRepo)
    monkeypatch.setattr(pipeline, "IntentHistoryRepository", _FakeIntentHistoryRepo)
    monkeypatch.setattr(pipeline, "RoutineTemplateRepository", _FakeRoutineRepo)
    monkeypatch.setattr(pipeline, "TaskEventRepository", _FakeTaskEventRepo)
    monkeypatch.setattr(pipeline, "TrelloClient", _FakeTrelloClient)
    monkeypatch.setattr(pipeline, "OnboardingService", _FakeOnboardingService)
    monkeypatch.setattr(pipeline, "TaskOrchestrator", _FakeTaskOrchestrator)


def test_board_url_is_linked_before_attachment_flow(monkeypatch):
    _setup(monkeypatch, _FakeClarificationRepo)
    _FakeTelegramClient.sent = []
    _FakeOnboardingService.called_with = []

    update = {
        "update_id": 999001,
        "message": {
            "from": {"id": 123},
            "chat": {"id": 456},
            "text": "https://trello.com/b/GMK1lbRp/work-board",
        },
    }
    _run(
        pipeline.process_telegram_update(
            db=object(),
            update=update,
            settings=_Settings(),
            openai_client=object(),
        )
    )

    assert _FakeOnboardingService.called_with == ["GMK1lbRp"]
    assert _FakeTelegramClient.sent


def test_board_url_has_priority_even_with_pending(monkeypatch):
    _setup(monkeypatch, _FakeClarificationRepoWithPending)
    _FakeTelegramClient.sent = []
    _FakeOnboardingService.called_with = []

    update = {
        "update_id": 999002,
        "message": {
            "from": {"id": 124},
            "chat": {"id": 457},
            "text": "https://trello.com/b/GMK1lbRp/work-board",
        },
    }
    _run(
        pipeline.process_telegram_update(
            db=object(),
            update=update,
            settings=_Settings(),
            openai_client=object(),
        )
    )

    assert _FakeOnboardingService.called_with == ["GMK1lbRp"]
    assert _FakeTelegramClient.sent
