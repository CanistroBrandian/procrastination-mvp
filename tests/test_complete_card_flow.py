"""Сценарии: пользователь сообщает о завершении ЦЕЛОЙ карточки (move в Done)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.db.models import Persona
from app.schemas.actions import AgentAction, AgentResult
from app.services.orchestrator import (
    TaskOrchestrator,
    _checklist_add_card_search_queries,
    _extract_index_from_reply,
    _infer_card_title_from_user_text,
)


def test_infer_card_title_from_guillemets():
    t = "В задачу, которая называется «Сделать уроки», нужно добавить предметы."
    assert _infer_card_title_from_user_text(t) == "Сделать уроки"


def test_infer_card_title_with_unicode_quotes_from_asr():
    t = "В задачу, которая называется \u201cСделать уроки\u201d, добавить предметы."
    assert _infer_card_title_from_user_text(t) == "Сделать уроки"


def test_checklist_search_queries_inferred_before_llm_card_name():
    action = AgentAction(
        action_type="create_checklist_item",
        card_name="Английский; Русский",
        checklist_item="Английский; Русский; География",
    )
    ut = "В задачу, которая называется «Сделать уроки», нужно добавить английский и русский."
    qs = _checklist_add_card_search_queries(ut, action)
    assert qs[0] == "Сделать уроки"


def test_add_checklist_resolves_card_by_title_in_text():
    board = _board_with([("c-lessons", "Сделать уроки", "INBOX", [])])
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="create_checklist_item",
                card_id=None,
                card_name=None,
                checklist_name="Шаги",
                checklist_item="Английский; Русский; География",
            ),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    out = _run(
        orch.process_text(
            profile,
            "В задачу, которая называется «Сделать уроки», нужно добавить английский, русский и географию.",
        )
    )
    cl_calls = [c for c in trello.calls if c[0] == "add_checklist"]
    item_calls = [c for c in trello.calls if c[0] == "add_check_item"]
    assert len(cl_calls) == 1
    assert cl_calls[0][1]["card_id"] == "c-lessons"
    assert len(item_calls) == 3
    assert out.text  # успешное выполнение без запроса ссылки на карточку


def test_add_checklist_resolves_despite_wrong_card_name_from_llm():
    """LLM иногда кладёт в card_name предметы чеклиста — поиск шёл по ним, а не по «Сделать уроки»."""
    board = _board_with([("c-lessons", "Сделать уроки", "INBOX", [])])
    agent = FakeAgent([
        AgentResult(
            action=AgentAction(
                action_type="create_checklist_item",
                card_id=None,
                card_name="Английский; Русский; География",
                checklist_name="Шаги",
                checklist_item="Английский; Русский; География",
            ),
            response_text="ok",
        ),
    ])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    _run(
        orch.process_text(
            profile,
            "В задачу, которая называется «Сделать уроки», нужно добавить английский, русский и географию.",
        )
    )
    cl_calls = [c for c in trello.calls if c[0] == "add_checklist"]
    assert len(cl_calls) == 1
    assert cl_calls[0][1]["card_id"] == "c-lessons"


def test_extract_index_from_reply_words_and_digits():
    assert _extract_index_from_reply("1", 5) == 1
    assert _extract_index_from_reply("номер 2", 5) == 2
    assert _extract_index_from_reply("Да, под номером один.", 5) == 1
    assert _extract_index_from_reply("выбираю первую", 5) == 1
    assert _extract_index_from_reply("второй вариант", 5) == 2
    assert _extract_index_from_reply("третий", 5) == 3
    assert _extract_index_from_reply("под номером один", 0) is None
    assert _extract_index_from_reply("семь", 5) is None
    assert _extract_index_from_reply("ничего из списка", 5) is None


def test_pick_card_candidate_understands_russian_word_index():
    candidates = [
        {"card_id": "c1", "card_name": "Сделать уроки"},
        {"card_id": "c2", "card_name": "Уроки: русский, английский и география"},
    ]
    chosen = TaskOrchestrator._pick_complete_card_candidate(candidates, "Да, под номером один.")
    assert chosen is not None and chosen["card_id"] == "c1"


def test_looks_like_new_command_for_index_replies_is_false():
    assert TaskOrchestrator._looks_like_new_command("Да, под номером один.") is False
    assert TaskOrchestrator._looks_like_new_command("номер два") is False
    assert TaskOrchestrator._looks_like_new_command("первую задачу") is False
    assert TaskOrchestrator._looks_like_new_command("создай задачу помыть машину") is True


def test_complete_card_pending_resumes_by_word_index():
    """Регресс: «Да, под номером один» в pending complete_card должен выбрать кандидата #1, а не идти в роутер."""
    board = _board_with(
        [
            ("card-1", "Сделать уроки", "DOING", []),
            ("card-2", "Уроки: русский, английский и география", "DOING", []),
        ]
    )
    agent = FakeAgent([])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)
    profile = FakeProfile()

    pending_action = AgentAction(
        action_type="ask_for_clarification",
        question="Нашёл несколько подходящих карточек для «уроки». Выберите номер:\n1) Сделать уроки\n2) Уроки: русский, английский и география",
        metadata={
            "after_clarification": "complete_card",
            "match_text": "уроки",
            "candidates": [
                {
                    "card_id": "card-1",
                    "card_name": "Сделать уроки",
                    "list_id": "DOING",
                    "incomplete_items": [],
                    "total_items": 0,
                },
                {
                    "card_id": "card-2",
                    "card_name": "Уроки: русский, английский и география",
                    "list_id": "DOING",
                    "incomplete_items": [],
                    "total_items": 0,
                },
            ],
        },
    )
    repo.store[profile.telegram_user_id] = FakePending(
        pending_action.model_dump_json(), pending_action.question or "",
    )

    out = _run(orch.process_text(profile, "Да, под номером один."))

    assert agent.received_texts == []  # LLM-роутер не должен дёргаться
    move_calls = [c for c in trello.calls if c[0] == "move_card"]
    assert len(move_calls) == 1
    assert move_calls[0][1]["card_id"] == "card-1"
    assert move_calls[0][1]["list_id"] == "DONE"
    assert "сделать уроки" in out.text.lower()


def test_parse_yes_no_latin_da_and_strip():
    assert TaskOrchestrator._parse_yes_no("Da, lobovoe pomili.") is True
    assert TaskOrchestrator._parse_yes_no("\ufeffДа.") is True


def test_implies_complete_card_force_yes_lobovoye():
    assert TaskOrchestrator._implies_complete_card_force_yes("Лобовую тоже помыли.") is True
    assert TaskOrchestrator._implies_complete_card_force_yes("просто текст") is False


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

    async def get_card_with_checklists(self, card_id: str) -> dict[str, Any]:
        self.calls.append(("get_card_with_checklists", {"card_id": card_id}))
        for c in self._board_payload:
            if c.get("id") == card_id:
                return c
        return {"id": card_id, "name": "", "checklists": []}

    async def list_card_checklists(self, card_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_card_checklists", {"card_id": card_id}))
        for c in self._board_payload:
            if c.get("id") == card_id:
                return list(c.get("checklists") or [])
        return []

    async def update_check_item_on_card(
        self, card_id: str, checklist_id: str, check_item_id: str, *, complete=None, name=None,
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
        for c in self._board_payload:
            if c.get("id") != card_id:
                continue
            for cl in c.get("checklists") or []:
                if cl.get("id") != checklist_id:
                    continue
                for ci in cl.get("checkItems") or []:
                    if ci.get("id") == check_item_id and complete is True:
                        ci["state"] = "complete"
        return {"id": check_item_id, "state": "complete" if complete else "incomplete"}

    async def move_card(self, card_id: str, list_id: str) -> dict[str, Any]:
        self.calls.append(("move_card", {"card_id": card_id, "list_id": list_id}))
        return {"id": card_id, "idList": list_id}

    async def add_checklist(self, card_id: str, name: str) -> dict[str, Any]:
        self.calls.append(("add_checklist", {"card_id": card_id, "name": name}))
        return {"id": f"cl-new-{card_id}", "name": name}

    async def add_check_item(self, checklist_id: str, name: str) -> dict[str, Any]:
        self.calls.append(("add_check_item", {"checklist_id": checklist_id, "name": name}))
        return {"id": f"ci-{len(self.calls)}", "name": name}


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
    assert any(c[0] == "list_card_checklists" for c in trello.calls)
    update_calls = [c for c in trello.calls if c[0] == "update_check_item_on_card"]
    assert len(update_calls) == 1
    assert update_calls[0][1]["check_item_id"] == "ci-glass"
    assert update_calls[0][1]["complete"] is True
    move_calls = [c for c in trello.calls if c[0] == "move_card"]
    assert len(move_calls) == 1
    assert move_calls[0][1]["list_id"] == "DONE"
    assert trello.calls.index(update_calls[0]) < trello.calls.index(move_calls[0])
    assert "отметил" in second.text.lower() or "выполнен" in second.text.lower()
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


def test_complete_card_force_without_explicit_da_lobovoye_still_completes_checklist():
    """Голос: «лобовую помыли» без слова «да» — всё равно закрываем пункты и переносим."""
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

    _run(orch.process_text(profile, "закрой машину"))
    assert profile.telegram_user_id in repo.store

    second = _run(orch.process_text(profile, "Лобовую тоже помыли."))
    update_calls = [c for c in trello.calls if c[0] == "update_check_item_on_card"]
    assert len(update_calls) == 1
    move_calls = [c for c in trello.calls if c[0] == "move_card"]
    assert len(move_calls) == 1
    assert "отметил" in second.text.lower() or "пункт" in second.text.lower()
