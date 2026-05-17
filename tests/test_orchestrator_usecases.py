"""End-to-end проверки use-case 1 (имя+срок обязательны) и use-case 2 (карточка+чеклист).

Тесты не требуют pytest-asyncio: корутины запускаются через asyncio.run.
"""

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
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._next_card_id = 100
        self._next_checklist_id = 200

    async def list_lists(self, board_id: str):
        self.calls.append(("list_lists", {"board_id": board_id}))
        return []

    async def list_board_cards_with_checklists(self, board_id: str):
        self.calls.append(("list_board_cards_with_checklists", {"board_id": board_id}))
        return []

    async def create_card(self, list_id, name, desc="", **kw):
        self._next_card_id += 1
        self.calls.append(("create_card", {"list_id": list_id, "name": name, "desc": desc, **kw}))
        return {"id": f"card{self._next_card_id}", "name": name}

    async def update_card(self, card_id, name=None, desc=None, **kw):
        self.calls.append(("update_card", {"card_id": card_id, "name": name, "desc": desc, **kw}))
        return {"id": card_id}

    async def delete_card(self, card_id):
        self.calls.append(("delete_card", {"card_id": card_id}))
        return {}

    async def move_card(self, card_id, list_id):
        self.calls.append(("move_card", {"card_id": card_id, "list_id": list_id}))
        return {"id": card_id}

    async def add_checklist(self, card_id, name):
        self._next_checklist_id += 1
        self.calls.append(("add_checklist", {"card_id": card_id, "name": name}))
        return {"id": f"cl{self._next_checklist_id}", "name": name}

    async def add_check_item(self, checklist_id, item_name):
        self.calls.append(("add_check_item", {"checklist_id": checklist_id, "item_name": item_name}))
        return {"id": "ci", "name": item_name}


class FakeAgent:
    def __init__(self, queued: list[AgentResult]):
        self.queued = list(queued)
        self.received_texts: list[str] = []

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):
        return []

    async def infer_action(
        self, *, text: str, persona: str = "mom", forced_intent: str | None = None,
    ) -> AgentResult:
        self.received_texts.append(text)
        if not self.queued:
            raise AssertionError("FakeAgent: нет заранее заданного результата")
        return self.queued.pop(0)


def _make_orch(agent: FakeAgent, trello: FakeTrello, repo: FakeClarificationRepo) -> TaskOrchestrator:
    return TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)


def _run(coro):
    return asyncio.run(coro)


def test_create_card_without_due_asks_clarification():
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(action_type="create_card", card_name="Помыть машину"),
            response_text="Ок",
        ),
    ])
    trello = FakeTrello()
    repo = FakeClarificationRepo()
    orch = _make_orch(agent, trello, repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "помыть машину"))

    assert "срок" in out.text.lower() or "сделать" in out.text.lower()
    assert trello.calls == []
    assert profile.telegram_user_id in repo.store


def test_create_card_with_due_calls_trello():
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="create_card",
                card_name="Помыть машину",
                due="2026-05-12T12:00:00+03:00",
            ),
            response_text="Готово",
        ),
    ])
    trello = FakeTrello()
    repo = FakeClarificationRepo()
    orch = _make_orch(agent, trello, repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "помыть машину 12 мая"))

    methods = [c[0] for c in trello.calls]
    assert methods == ["create_card"]
    assert out.text == "Готово"
    assert profile.telegram_user_id not in repo.store


def test_create_card_due_inferred_from_text_when_model_misses_it():
    """Use-case 1 fallback: модель не дала due, но в тексте «завтра» — оркестратор подставит."""
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(action_type="create_card", card_name="Помыть машину"),
            response_text="Готово",
        ),
    ])
    trello = FakeTrello()
    repo = FakeClarificationRepo()
    orch = _make_orch(agent, trello, repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "помыть машину завтра"))

    methods = [c[0] for c in trello.calls]
    assert methods.count("create_card") == 1
    create_kwargs = trello.calls[0][1]
    assert create_kwargs["due"] is not None and "T12:00:00" in create_kwargs["due"]
    assert out.text == "Готово"


def test_create_card_with_inline_checklist_creates_card_and_items():
    """Use-case 2: одна фраза → карточка + чеклист с пунктами."""
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="create_card",
                card_name="Убраться в квартире",
                due="2026-05-12T12:00:00+03:00",
                checklist_name="Шаги",
                checklist_item="Пропылесосить; Помыть окна",
            ),
            response_text="Создал задачу с чеклистом",
        ),
    ])
    trello = FakeTrello()
    repo = FakeClarificationRepo()
    orch = _make_orch(agent, trello, repo)
    profile = FakeProfile()

    _run(orch.process_text(profile, "убраться в квартире, пропылесосить и помыть окна, до завтра"))

    methods = [c[0] for c in trello.calls]
    assert methods == ["create_card", "add_checklist", "add_check_item", "add_check_item"]
    items = [c[1]["item_name"] for c in trello.calls if c[0] == "add_check_item"]
    assert items == ["Пропылесосить", "Помыть окна"]


def test_clarification_resume_for_create_card_completes_creation():
    """Сначала уточняем срок; ответ пользователя порождает create_card с due."""
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(action_type="create_card", card_name="Помыть машину"),
            response_text="Уточню срок",
        ),
        AgentResult(
            action=AgentAction(
                action_type="create_card",
                card_name="Помыть машину",
                due="2026-05-13T12:00:00+03:00",
            ),
            response_text="Готово",
        ),
    ])
    trello = FakeTrello()
    repo = FakeClarificationRepo()
    orch = _make_orch(agent, trello, repo)
    profile = FakeProfile()

    first_reply = _run(orch.process_text(profile, "помыть машину"))
    assert "срок" in first_reply.text.lower() or "сделать" in first_reply.text.lower()
    assert trello.calls == []

    second_reply = _run(orch.process_text(profile, "13 мая в 12:00"))
    assert second_reply.text == "Готово"
    methods = [c[0] for c in trello.calls]
    assert methods == ["create_card"]
    assert profile.telegram_user_id not in repo.store


def test_create_card_clarification_abort_voice_phrase_does_not_call_llm():
    """После уточнения срока: «не надо ставить задачу» — отмена без второго вызова агента."""
    agent = FakeAgent(
        [
            AgentResult(
                action=AgentAction(action_type="create_card", card_name="Что-то"),
                response_text="Уточню",
            ),
        ],
    )
    trello = FakeTrello()
    repo = FakeClarificationRepo()
    orch = _make_orch(agent, trello, repo)
    profile = FakeProfile()

    _run(orch.process_text(profile, "поставь задачу"))
    assert trello.calls == []

    out = _run(orch.process_text(profile, "Не надо ставить задачу."))
    assert "отмен" in out.text.lower()
    assert trello.calls == []
    assert agent.received_texts == ["поставь задачу"]
    assert profile.telegram_user_id not in repo.store


def test_abort_create_card_phrases():
    from app.services.orchestrator import TaskOrchestrator

    assert TaskOrchestrator._looks_like_abort_create_card_clarification("Не надо ставить задачу.")
    assert TaskOrchestrator._looks_like_abort_create_card_clarification("отмена")
    assert TaskOrchestrator._looks_like_abort_create_card_clarification("нет")
    assert not TaskOrchestrator._looks_like_abort_create_card_clarification("13 мая в 12:00")


def test_pure_checklist_request_without_card_id_asks_for_card():
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="create_checklist_item",
                checklist_item="Шаг1; Шаг2",
            ),
            response_text="Нужна карточка",
        ),
    ])
    trello = FakeTrello()
    repo = FakeClarificationRepo()
    orch = _make_orch(agent, trello, repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "добавь пункты в чеклист"))
    assert "карточк" in out.text.lower()
    methods = [c[0] for c in trello.calls]
    assert methods == ["list_lists", "list_board_cards_with_checklists"]
