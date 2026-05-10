"""Сценарии: пользователь сообщает о завершении ЦЕЛОЙ карточки (move в Done)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.db.models import Persona
from app.schemas.actions import AgentAction, AgentResult
from app.services.orchestrator import TaskOrchestrator


@dataclass
class FakeProfile:
    telegram_user_id: int = 1
    persona: Persona = Persona.MOM
    trello_board_id: str | None = "BOARD"
    trello_inbox_list_id: str | None = "INBOX"
    trello_doing_list_id: str | None = "DOING"
    trello_done_list_id: str | None = "DONE"


class FakePending:
    def __init__(self, draft_action_json: str, question: str = ""):
        self.draft_action_json = draft_action_json
        self.question = question


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
    def __init__(self, board_payload: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._board_payload = board_payload or []

    async def list_lists(self, board_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_lists", {"board_id": board_id}))
        lids = sorted({str(c.get("idList") or "") for c in self._board_payload if c.get("idList")})
        return [{"id": lid, "name": lid, "closed": False} for lid in lids if lid]

    async def list_board_cards_with_checklists(self, board_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_board_cards_with_checklists", {"board_id": board_id}))
        return self._board_payload

    async def move_card(self, card_id: str, list_id: str) -> dict[str, Any]:
        self.calls.append(("move_card", {"card_id": card_id, "list_id": list_id}))
        return {"id": card_id, "idList": list_id}


class FakeAgent:
    def __init__(self, queued: list[AgentResult]):
        self.queued = list(queued)
        self.received_texts: list[str] = []

    async def infer_action(
        self, *, text: str, persona: str = "mom", forced_intent: str | None = None,
    ) -> AgentResult:
        self.received_texts.append(text)
        if not self.queued:
            raise AssertionError("FakeAgent: пустая очередь")
        return self.queued.pop(0)

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):
        # По умолчанию семантический поиск возвращает пустой список — fuzzy решает.
        return []


def _board_with(
    cards: list[tuple[str, str, str, list[tuple[str, str, str]]]],
) -> list[dict[str, Any]]:
    """cards: [(card_id, card_name, list_id, [(check_item_id, item_name, state)])]"""
    out = []
    for c_id, c_name, list_id, items in cards:
        out.append(
            {
                "id": c_id,
                "name": c_name,
                "idList": list_id,
                "closed": False,
                "checklists": [
                    {
                        "id": f"cl-{c_id}",
                        "name": "Шаги",
                        "checkItems": [
                            {"id": ci_id, "name": ci_name, "state": st}
                            for ci_id, ci_name, st in items
                        ],
                    }
                ]
                if items
                else [],
            }
        )
    return out


def _run(coro):
    return asyncio.run(coro)


def test_complete_card_all_done_moves_to_done_list():
    board = _board_with(
        [
            (
                "card-car",
                "Обслуживание машины",
                "DOING",
                [
                    ("ci-glass", "Помыть лобовое стекло", "complete"),
                    ("ci-vacuum", "Пропылесосить в машине", "complete"),
                ],
            ),
        ]
    )
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="complete_card_by_text", match_text="обслуживание машины",
            ),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "перевести обслуживание машины в завершённые"))

    move_calls = [c for c in trello.calls if c[0] == "move_card"]
    assert len(move_calls) == 1
    assert move_calls[0][1] == {"card_id": "card-car", "list_id": "DONE"}
    assert "обслуживание машины" in out.text.lower()
    assert "завершённые" in out.text.lower() or "завершенные" in out.text.lower()
    assert profile.telegram_user_id not in repo.store


def test_complete_card_with_incomplete_items_asks_force_then_yes():
    board = _board_with(
        [
            (
                "card-car",
                "Обслуживание машины",
                "DOING",
                [
                    ("ci-glass", "Помыть лобовое стекло", "incomplete"),
                    ("ci-vacuum", "Пропылесосить в машине", "complete"),
                ],
            ),
        ]
    )
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="complete_card_by_text", match_text="обслуживание машины",
            ),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    first = _run(orch.process_text(profile, "обслуживание машины — закрыть"))
    assert "невыполненные" in first.text.lower() or "помыть лобовое" in first.text.lower()
    assert profile.telegram_user_id in repo.store
    assert all(c[0] != "move_card" for c in trello.calls)

    second = _run(orch.process_text(profile, "да"))
    move_calls = [c for c in trello.calls if c[0] == "move_card"]
    assert len(move_calls) == 1
    assert move_calls[0][1]["list_id"] == "DONE"
    assert "обслуживание машины" in second.text.lower()


def test_complete_card_with_incomplete_items_force_no_keeps_card():
    board = _board_with(
        [
            (
                "card-car",
                "Обслуживание машины",
                "DOING",
                [("ci", "Помыть лобовое стекло", "incomplete")],
            ),
        ]
    )
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="complete_card_by_text", match_text="обслуживание машины",
            ),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    _run(orch.process_text(profile, "обслуживание машины — закрыть"))
    out = _run(orch.process_text(profile, "нет"))

    assert all(c[0] != "move_card" for c in trello.calls)
    assert "не перевод" in out.text.lower() or "сначала отмет" in out.text.lower()


def test_complete_card_ambiguous_asks_to_choose_then_resumes_by_number():
    board = _board_with(
        [
            ("card-a", "Отчёт по проекту А", "DOING", []),
            ("card-b", "Отчёт по проекту Б", "DOING", []),
        ]
    )
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(action_type="complete_card_by_text", match_text="отчёт"),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    first = _run(orch.process_text(profile, "закрой отчёт"))
    assert "выберите номер" in first.text.lower() or "не уверен" in first.text.lower()
    assert profile.telegram_user_id in repo.store
    assert all(c[0] != "move_card" for c in trello.calls)

    second = _run(orch.process_text(profile, "2"))
    move_calls = [c for c in trello.calls if c[0] == "move_card"]
    assert len(move_calls) == 1
    assert move_calls[0][1]["card_id"] == "card-b"
    assert "проекту б" in second.text.lower()


def test_complete_card_no_match_asks_for_clarification():
    board = _board_with([("c", "Совсем другая задача", "DOING", [])])
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="complete_card_by_text", match_text="обслуживание машины",
            ),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "закрой обслуживание машины"))
    assert "не нашёл" in out.text.lower() or "уточните" in out.text.lower()
    assert all(c[0] != "move_card" for c in trello.calls)
    assert profile.telegram_user_id in repo.store


def test_complete_card_skips_cards_already_in_done():
    board = _board_with(
        [
            ("card-old", "Обслуживание машины", "DONE", []),
            ("card-active", "Обслуживание машины", "DOING", []),
        ]
    )
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="complete_card_by_text", match_text="обслуживание машины",
            ),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    _run(orch.process_text(profile, "закрой обслуживание машины"))
    move_calls = [c for c in trello.calls if c[0] == "move_card"]
    assert len(move_calls) == 1
    assert move_calls[0][1]["card_id"] == "card-active"
