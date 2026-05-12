from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from app.db.models import Persona
from app.schemas.actions import AgentAction, AgentResult
from app.services.orchestrator import TaskOrchestrator


@dataclass
class FakeProfile:
    telegram_user_id: int = 7
    persona: Persona = Persona.MOM
    trello_board_id: str | None = "BOARD"
    trello_inbox_list_id: str | None = "INBOX"
    trello_doing_list_id: str | None = "DOING"
    trello_done_list_id: str | None = "DONE"


class FakeClarificationRepo:
    async def get(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def clear(self, telegram_user_id: int) -> None:  # noqa: ARG002
        return None

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str) -> None:  # noqa: ARG002
        return None


class FakeConversationStateRepo:
    def __init__(self) -> None:
        self.by_user: dict[int, dict[str, Any]] = {}

    async def get(self, telegram_user_id: int):
        data = self.by_user.get(telegram_user_id)
        if data is None:
            return None
        return SimpleNamespace(**data)

    async def upsert(self, telegram_user_id: int, **kwargs):  # type: ignore[no-untyped-def]
        cur = self.by_user.get(
            telegram_user_id,
            {
                "telegram_user_id": telegram_user_id,
                "active_flow": None,
                "active_card_id": None,
                "active_card_name": None,
                "pending_action_json": None,
                "missing_slots_json": None,
                "candidate_cards_json": None,
                "candidate_items_json": None,
                "flow_id": None,
            },
        )
        cur.update(kwargs)
        self.by_user[telegram_user_id] = cur
        return SimpleNamespace(**cur)


class FakeTrello:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cards = [
            {"id": "card-created", "name": "План разработки приложения", "idList": "INBOX", "closed": False, "checklists": []},
        ]

    async def create_card(self, list_id, name, desc="", **kw):  # type: ignore[no-untyped-def]
        self.calls.append(("create_card", {"list_id": list_id, "name": name, "desc": desc, **kw}))
        card = {"id": "card-created", "name": name}
        return card

    async def update_card(self, card_id, name=None, desc=None, **kw):  # type: ignore[no-untyped-def]
        self.calls.append(("update_card", {"card_id": card_id, "name": name, "desc": desc, **kw}))
        return {"id": card_id}

    async def delete_card(self, card_id):  # type: ignore[no-untyped-def]
        self.calls.append(("delete_card", {"card_id": card_id}))
        return {"id": card_id}

    async def list_lists(self, board_id: str):  # noqa: ARG002
        return [{"id": "INBOX", "name": "INBOX", "closed": False}]

    async def list_board_cards_with_checklists(self, board_id: str):  # noqa: ARG002
        return self.cards


class FakeAgent:
    def __init__(self, queued: list[AgentResult]) -> None:
        self.queued = list(queued)

    async def infer_action(self, *, text: str, persona: str = "mom", forced_intent: str | None = None, intent_context: str | None = None):  # noqa: ARG002
        if not self.queued:
            raise AssertionError("queue is empty")
        return self.queued.pop(0)

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):  # noqa: ARG002
        return []


def _run(coro):
    return asyncio.run(coro)


def test_conversation_state_is_used_for_rename_without_card_id():
    state_repo = FakeConversationStateRepo()
    agent = FakeAgent(
        [
            AgentResult(
                action=AgentAction(action_type="create_card", card_name="План разработки приложения", due="2026-05-13T10:00:00+03:00"),
                response_text="Создал задачу.",
            ),
            AgentResult(
                action=AgentAction(action_type="update_card", card_name="План по захвату мира"),
                response_text="Переименовал.",
            ),
        ],
    )
    trello = FakeTrello()
    orch = TaskOrchestrator(
        agent_service=agent,
        trello_client=trello,
        clarification_repo=FakeClarificationRepo(),
        conversation_state_repo=state_repo,
    )
    profile = FakeProfile()

    _run(orch.process_text(profile, "Поставь задачу на завтра: План разработки приложения"))
    _run(orch.process_text(profile, "Переименуй задачу на «План по захвату мира»."))

    update_calls = [c for c in trello.calls if c[0] == "update_card"]
    assert len(update_calls) == 1
    assert update_calls[0][1]["card_id"] == "card-created"


def test_delete_uses_active_card_from_conversation_state():
    state_repo = FakeConversationStateRepo()
    state_repo.by_user[7] = {
        "telegram_user_id": 7,
        "active_flow": None,
        "active_card_id": "card-created",
        "active_card_name": "План разработки приложения",
        "pending_action_json": None,
        "missing_slots_json": None,
        "candidate_cards_json": None,
        "candidate_items_json": None,
        "flow_id": "flow-1",
    }
    agent = FakeAgent(
        [
            AgentResult(
                action=AgentAction(action_type="delete_card"),
                response_text="Удалил.",
            ),
        ],
    )
    trello = FakeTrello()
    orch = TaskOrchestrator(
        agent_service=agent,
        trello_client=trello,
        clarification_repo=FakeClarificationRepo(),
        conversation_state_repo=state_repo,
    )
    profile = FakeProfile()

    _run(orch.process_text(profile, "Удали эту задачу"))

    delete_calls = [c for c in trello.calls if c[0] == "delete_card"]
    assert len(delete_calls) == 1
    assert delete_calls[0][1]["card_id"] == "card-created"
