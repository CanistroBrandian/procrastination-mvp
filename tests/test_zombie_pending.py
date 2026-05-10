"""Регрессионные тесты на «зомби-pending»: pending от прошлой неудачной попытки
не должен бесконечно перехватывать новые команды пользователя.

Сценарий, который мы видели в продакшене:
1) Пользователь сказал «я помыл окна», бот не нашёл пункта → сохранил pending complete_task.
2) Пользователь сказал новую команду «давай поставим задачу делать уроки».
3) Старый код ВНОВЬ интерпретировал новую фразу как ответ на уточнение чеклиста
   и снова падал в «не нашёл активного пункта» → бесконечный цикл.

Здесь проверяем, что новая команда отпускается в LLM-роутер.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.db.models import Persona
from app.schemas.actions import AgentAction, AgentResult
from app.services.orchestrator import TaskOrchestrator


def _run(coro):
    return asyncio.run(coro)


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

    async def create_card(  # type: ignore[no-untyped-def]
        self,
        list_id,
        name,
        description="",
        *,
        due=None,
        start=None,
        due_complete=None,
        position=None,
    ):
        self.calls.append((
            "create_card",
            {
                "list_id": list_id,
                "name": name,
                "description": description,
                "due": due,
                "start": start,
                "due_complete": due_complete,
                "position": position,
            },
        ))
        return {"id": "new-card-id"}

    async def add_checklist(self, card_id: str, name: str):
        self.calls.append(("add_checklist", {"card_id": card_id, "name": name}))
        return {"id": "cl-id"}

    async def add_check_item(self, checklist_id: str, name: str):
        self.calls.append(("add_check_item", {"checklist_id": checklist_id, "name": name}))
        return {"id": "ci-id"}


class FakeAgent:
    def __init__(self, queued: list[AgentResult]):
        self.queued = list(queued)
        self.received_texts: list[str] = []
        self.received_intents: list[str | None] = []

    async def infer_action(
        self, *, text: str, persona: str = "mom", forced_intent: str | None = None,
    ) -> AgentResult:
        self.received_texts.append(text)
        self.received_intents.append(forced_intent)
        if not self.queued:
            raise AssertionError("FakeAgent: пустая очередь")
        return self.queued.pop(0)

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):
        return []


def _set_pending(repo: FakeClarificationRepo, profile: FakeProfile, draft: dict[str, Any]) -> None:
    repo.store[profile.telegram_user_id] = FakePending(json.dumps(draft, ensure_ascii=False))


def test_zombie_complete_task_pending_is_dropped_for_new_command():
    """В БД лежит pending после неудачного complete_task без candidates.
    Пользователь шлёт новую команду «поставь задачу...» — должны вызвать LLM-роутер."""
    profile = FakeProfile()
    repo = FakeClarificationRepo()
    _set_pending(
        repo,
        profile,
        {
            "action_type": "ask_for_clarification",
            "metadata": {
                "after_clarification": "complete_task",
                "match_text": "помыл окна",
                # candidates пуст — это «не нашёл пункт» сценарий.
            },
        },
    )

    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="create_card",
                card_name="Сделать уроки",
                due="2026-05-11T09:00:00.000Z",
            ),
            response_text="Создаю.",
        ),
    ])
    trello = FakeTrello()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)

    out = _run(orch.process_text(profile, "Давай поставим задачу делать уроки"))

    assert agent.received_texts == ["Давай поставим задачу делать уроки"]
    assert any(c[0] == "create_card" for c in trello.calls)
    assert "уроки" in out.text.lower() or "созда" in out.text.lower() or "записал" in out.text.lower() or "принято" in out.text.lower()
    # Pending не должен снова появиться.
    assert profile.telegram_user_id not in repo.store


def test_zombie_complete_card_pending_is_dropped_for_new_command():
    """То же самое, но для complete_card."""
    profile = FakeProfile()
    repo = FakeClarificationRepo()
    _set_pending(
        repo,
        profile,
        {
            "action_type": "ask_for_clarification",
            "metadata": {
                "after_clarification": "complete_card",
                "match_text": "обслуживание машины",
            },
        },
    )

    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="create_card",
                card_name="Купить молоко",
                due="2026-05-11T18:00:00.000Z",
            ),
            response_text="Создаю.",
        ),
    ])
    trello = FakeTrello()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)

    _run(orch.process_text(profile, "Поставь задачу купить молоко на завтра"))

    assert any(c[0] == "create_card" for c in trello.calls)
    assert profile.telegram_user_id not in repo.store


def test_complete_task_with_candidates_still_resolves_short_reply():
    """Если есть candidates и пользователь пишет короткий ответ ('1'), pending всё ещё работает."""
    profile = FakeProfile()
    repo = FakeClarificationRepo()
    _set_pending(
        repo,
        profile,
        {
            "action_type": "ask_for_clarification",
            "metadata": {
                "after_clarification": "complete_task",
                "candidates": [
                    {
                        "card_id": "c1",
                        "card_name": "A",
                        "checklist_id": "cl1",
                        "check_item_id": "ci1",
                        "item_name": "Помыть окна",
                    },
                    {
                        "card_id": "c2",
                        "card_name": "B",
                        "checklist_id": "cl2",
                        "check_item_id": "ci2",
                        "item_name": "Помыть пол",
                    },
                ],
            },
        },
    )

    class T(FakeTrello):
        async def update_check_item_on_card(
            self, card_id, checklist_id, check_item_id, *, complete=None, name=None,
        ):
            self.calls.append((
                "update_check_item_on_card",
                {"card_id": card_id, "checklist_id": checklist_id, "check_item_id": check_item_id, "complete": complete},
            ))
            return {"id": check_item_id}

    agent = FakeAgent([])  # не должен быть вызван
    trello = T()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)

    _run(orch.process_text(profile, "2"))

    upd_calls = [c for c in trello.calls if c[0] == "update_check_item_on_card"]
    assert len(upd_calls) == 1
    assert upd_calls[0][1]["check_item_id"] == "ci2"
    assert agent.received_texts == []


def test_looks_like_new_command_heuristic():
    """Простая проверка эвристики на разных входах."""
    fn = TaskOrchestrator._looks_like_new_command

    assert fn("Давай поставим задачу делать уроки") is True
    assert fn("Поставь задачу купить хлеб") is True
    assert fn("Создай карточку") is True
    assert fn("Удали карточку посадить дерево") is True
    assert fn("Перенеси на пятницу") is True

    assert fn("1") is False
    assert fn("2") is False
    assert fn("да") is False
    assert fn("нет") is False
    assert fn("Помыл окна") is False
    assert fn("Помыл лобовое стекло") is False
