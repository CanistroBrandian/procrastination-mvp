from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.db.models import Persona
from app.schemas.actions import AgentAction
from app.schemas.actions import AgentResult
from app.services.orchestrator import TaskOrchestrator
from tests.test_complete_card_flow import FakePending


@dataclass
class FakeProfile:
    telegram_user_id: int = 1
    persona: Persona = Persona.MOM
    trello_board_id: str | None = "BOARD"
    trello_inbox_list_id: str | None = "INBOX"
    trello_doing_list_id: str | None = "DOING"
    trello_done_list_id: str | None = "DONE"


class FakeClarificationRepo:
    def __init__(self) -> None:
        self.store: dict[int, FakePending] = {}

    async def get(self, telegram_user_id: int):
        return self.store.get(telegram_user_id)

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str) -> None:
        self.store[telegram_user_id] = FakePending(draft_action_json, question)

    async def clear(self, telegram_user_id: int) -> None:
        self.store.pop(telegram_user_id, None)


class FakeTrello:
    def __init__(self, board_payload: list[dict[str, Any]]) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._board_payload = board_payload

    async def list_lists(self, board_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_lists", {"board_id": board_id}))
        lids = sorted({str(c.get("idList") or "") for c in self._board_payload if c.get("idList")})
        return [{"id": lid, "name": lid, "closed": False} for lid in lids if lid]

    async def list_board_cards_with_checklists(self, board_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_board_cards_with_checklists", {"board_id": board_id}))
        return self._board_payload

    async def update_card(self, card_id, name=None, desc=None, **kw):  # type: ignore[no-untyped-def]
        self.calls.append(("update_card", {"card_id": card_id, "name": name, "desc": desc, **kw}))
        return {"id": card_id}


class FakeAgent:
    def __init__(self, queued: list[AgentResult]) -> None:
        self.queued = list(queued)
        self.received_texts: list[str] = []

    async def infer_action(self, *, text: str, persona: str = "mom", forced_intent: str | None = None) -> AgentResult:
        self.received_texts.append(text)
        if not self.queued:
            raise AssertionError("FakeAgent: empty queue")
        return self.queued.pop(0)

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):  # noqa: ARG002
        return []


def _run(coro):
    return asyncio.run(coro)


def _board_with_one() -> list[dict[str, Any]]:
    return [
        {
            "id": "c-plan",
            "name": "План разработки приложения",
            "idList": "DOING",
            "closed": False,
            "checklists": [],
        },
    ]


def test_update_card_rename_resolves_card_by_old_name_without_card_id():
    agent = FakeAgent(
        [
            AgentResult(
                action=AgentAction(
                    action_type="update_card",
                    card_name="План по захвату мира",
                ),
                response_text="Ок",
            ),
        ],
    )
    trello = FakeTrello(_board_with_one())
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    _run(
        orch.process_text(
            profile,
            "Давай переименуем задачу «План разработки приложения» на название «План по захвату мира».",
        ),
    )

    update_calls = [c for c in trello.calls if c[0] == "update_card"]
    assert len(update_calls) == 1
    assert update_calls[0][1]["card_id"] == "c-plan"
    assert update_calls[0][1]["name"] == "План по захвату мира"
    assert profile.telegram_user_id not in repo.store


def test_update_card_pending_pick_card_resumes_on_card_name_reply():
    agent = FakeAgent([])
    trello = FakeTrello(_board_with_one())
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    pending_action = AgentAction(
        action_type="ask_for_clarification",
        question="Уточните, какую карточку нужно изменить.",
        metadata={
            "after_clarification": "update_card_pick_card",
            "pending_update_card_name": "План по захвату мира",
        },
    )
    repo.store[profile.telegram_user_id] = FakePending(
        pending_action.model_dump_json(),
        question=pending_action.question or "",
    )

    out = _run(orch.process_text(profile, "И задача называется «План разработки приложения»."))

    update_calls = [c for c in trello.calls if c[0] == "update_card"]
    assert len(update_calls) == 1
    assert update_calls[0][1]["card_id"] == "c-plan"
    assert update_calls[0][1]["name"] == "План по захвату мира"
    assert agent.received_texts == []
    assert "обновил" in out.text.lower()
