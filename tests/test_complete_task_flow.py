"""Сценарии: пользователь сообщает о выполненной подзадаче — ищем по чеклистам."""

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

    async def update_check_item_on_card(
        self, card_id, checklist_id, check_item_id, *, complete=None, name=None,
    ):
        self.calls.append(
            (
                "update_check_item_on_card",
                {
                    "card_id": card_id,
                    "checklist_id": checklist_id,
                    "check_item_id": check_item_id,
                    "complete": complete,
                    "name": name,
                },
            )
        )
        return {"id": check_item_id, "state": "complete" if complete else "incomplete"}


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
        return []


def _board_with(cards: list[tuple[str, str, list[tuple[str, str, str]]]]) -> list[dict[str, Any]]:
    """cards: [(card_id, card_name, [(check_item_id, item_name, state)])]"""
    out = []
    for c_id, c_name, items in cards:
        out.append(
            {
                "id": c_id,
                "name": c_name,
                "idList": "L",
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
                ],
            }
        )
    return out


def _run(coro):
    return asyncio.run(coro)


def test_complete_task_unique_match_marks_done():
    board = _board_with(
        [
            (
                "card-clean",
                "Убраться в квартире",
                [
                    ("ci-windows", "Помыть окна", "incomplete"),
                    ("ci-vacuum", "Пропылесосить", "incomplete"),
                ],
            ),
        ]
    )
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(action_type="complete_task_from_text", match_text="помыл окна"),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "я помыл окна"))

    methods = [c[0] for c in trello.calls]
    assert "update_check_item_on_card" in methods
    upd = next(c for c in trello.calls if c[0] == "update_check_item_on_card")[1]
    assert upd["check_item_id"] == "ci-windows"
    assert upd["complete"] is True
    assert "помыть окна" in out.text.lower()
    assert profile.telegram_user_id not in repo.store


def test_complete_task_ambiguous_asks_to_choose_then_resumes_by_number():
    board = _board_with(
        [
            ("card-a", "Отчёт по проекту А", [("ci-a", "Сделать отчёт", "incomplete")]),
            ("card-b", "Отчёт по проекту Б", [("ci-b", "Сделать отчёт", "incomplete")]),
        ]
    )
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(action_type="complete_task_from_text", match_text="сделал отчёт"),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    first = _run(orch.process_text(profile, "сделал отчёт"))
    assert "выберите номер" in first.text.lower() or "не уверен" in first.text.lower()
    assert profile.telegram_user_id in repo.store
    assert all(c[0] != "update_check_item_on_card" for c in trello.calls)

    second = _run(orch.process_text(profile, "2"))
    upd_calls = [c for c in trello.calls if c[0] == "update_check_item_on_card"]
    assert len(upd_calls) == 1
    assert upd_calls[0][1]["check_item_id"] == "ci-b"
    assert upd_calls[0][1]["complete"] is True
    assert "проекту б" in second.text.lower() or "отчёт" in second.text.lower()


def test_complete_task_no_matches_asks_for_clarification():
    board = _board_with(
        [
            ("c", "Карточка", [("ci", "Купить хлеб", "incomplete")]),
        ]
    )
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(action_type="complete_task_from_text", match_text="починил машину"),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "починил машину"))
    assert "не нашёл" in out.text.lower() or "уточните" in out.text.lower()
    assert all(c[0] != "update_check_item_on_card" for c in trello.calls)
    assert profile.telegram_user_id in repo.store
