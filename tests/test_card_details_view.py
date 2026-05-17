from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.db.models import Persona
from app.services.orchestrator import TaskOrchestrator


@dataclass
class FakeProfile:
    telegram_user_id: int = 501
    persona: Persona = Persona.MOM
    trello_board_id: str | None = "BOARD"
    trello_inbox_list_id: str | None = "INBOX"
    trello_doing_list_id: str | None = "DOING"
    trello_done_list_id: str | None = "DONE"


class FakeClarificationRepo:
    async def get(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def clear(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str):  # noqa: ARG002
        return None


class FakeConversationStateRepo:
    def __init__(self) -> None:
        self.state: dict[int, dict[str, Any]] = {}

    async def get(self, telegram_user_id: int):
        row = self.state.get(telegram_user_id)
        if row is None:
            return None

        class _State:
            active_flow = row.get("active_flow")
            active_card_id = row.get("active_card_id")
            active_card_name = row.get("active_card_name")

        return _State()

    async def upsert(self, telegram_user_id: int, **kwargs):
        current = dict(self.state.get(telegram_user_id) or {})
        current.update(kwargs)
        self.state[telegram_user_id] = current
        return None


class FakeTrello:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_lists(self, board_id: str):  # noqa: ARG002
        self.calls.append(("list_lists", {}))
        return [{"id": "DOING", "name": "Doing", "closed": False}]

    async def list_board_cards_with_checklists(self, board_id: str):  # noqa: ARG002
        self.calls.append(("list_board_cards_with_checklists", {}))
        return [
            {
                "id": "c-home",
                "name": "Дела по дому",
                "idList": "DOING",
                "closed": False,
                "shortUrl": "",
                "checklists": [],
            },
        ]

    async def list_card_checklists(self, card_id: str):  # noqa: ARG002
        self.calls.append(("list_card_checklists", {"card_id": card_id}))
        return [
            {
                "id": "cl-1",
                "name": "Покупки",
                "checkItems": [
                    {"id": "i-1", "name": "Купить муку", "state": "incomplete"},
                    {"id": "i-2", "name": "Купить лук", "state": "complete"},
                ],
            }
        ]

    async def get_card(self, card_id: str):  # noqa: ARG002
        self.calls.append(("get_card", {"card_id": card_id}))
        return {
            "id": card_id,
            "name": "Дела по дому",
            "desc": "Полезные ссылки: https://example.com/spec and https://example.com/video",
            "due": "2026-05-15T13:00:00+03:00",
            "dueComplete": False,
        }

    async def list_card_attachments(self, card_id: str):  # noqa: ARG002
        self.calls.append(("list_card_attachments", {"card_id": card_id}))
        return [
            {"name": "brief.pdf", "url": "https://files.example/brief.pdf", "isUpload": True},
            {"name": "external link", "url": "https://example.com", "isUpload": False},
        ]


class FakeAgent:
    async def infer_action(self, **kwargs):  # noqa: ARG002
        raise AssertionError("LLM path should not be used for card view queries")

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):  # noqa: ARG002
        return []


def _run(coro):
    return asyncio.run(coro)


def test_view_checklist_by_card_name_updates_active_card_context():
    profile = FakeProfile()
    state_repo = FakeConversationStateRepo()
    orch = TaskOrchestrator(
        agent_service=FakeAgent(),
        trello_client=FakeTrello(),
        clarification_repo=FakeClarificationRepo(),
        conversation_state_repo=state_repo,
    )

    out = _run(orch.process_text(profile, "А какие задачи внутри \"Дела по дому\"?"))

    assert out.action_type == "show_card_details"
    assert out.resolved_card_id == "c-home"
    assert "Купить муку" in out.text
    saved = state_repo.state.get(profile.telegram_user_id)
    assert saved is not None
    assert saved.get("active_card_id") == "c-home"


def test_follow_up_queries_use_active_card_context_without_name():
    profile = FakeProfile()
    state_repo = FakeConversationStateRepo()
    state_repo.state[profile.telegram_user_id] = {
        "active_flow": None,
        "active_card_id": "c-home",
        "active_card_name": "Дела по дому",
        "pending_action_json": None,
    }
    trello = FakeTrello()
    orch = TaskOrchestrator(
        agent_service=FakeAgent(),
        trello_client=trello,
        clarification_repo=FakeClarificationRepo(),
        conversation_state_repo=state_repo,
    )

    due = _run(orch.process_text(profile, "Когда нужно завершить задачу?"))
    links = _run(orch.process_text(profile, "Покажи какие ссылки есть в задании"))
    files = _run(orch.process_text(profile, "Покажи файлы"))

    assert due.action_type == "show_card_details"
    assert "2026-05-15T13:00:00+03:00" in due.text
    assert "https://example.com/spec" in links.text
    assert "brief.pdf" in files.text
    assert "https://files.example/brief.pdf" in files.text
    assert any(call[0] == "list_card_attachments" for call in trello.calls)
