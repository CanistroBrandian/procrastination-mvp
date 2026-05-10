"""Гибридный поиск карточек: fuzzy + семантический LLM-агент.

Проверяем:
1) Если fuzzy уверенно нашёл одну карточку с высоким score — LLM не вызывается.
2) Если fuzzy слабый или ничего не нашёл — вызываем LLM-семантику; merge сводит
   результаты и оркестратор предлагает кандидата.
3) Если оба источника дают несколько кандидатов с близкими score — оркестратор
   спрашивает пользователя номер.
4) Невалидные id от LLM игнорируются.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.db.models import Persona
from app.schemas.actions import AgentAction, AgentResult
from app.services.card_search import CardSearchHit, merge_search_results
from app.services.checklist_match import CardRef
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

    async def move_card(self, card_id: str, list_id: str) -> dict[str, Any]:
        self.calls.append(("move_card", {"card_id": card_id, "list_id": list_id}))
        return {"id": card_id, "idList": list_id}


@dataclass
class FakeAgent:
    intent_results: list[AgentResult] = field(default_factory=list)
    semantic_hits: list[CardSearchHit] = field(default_factory=list)
    semantic_calls: int = 0
    received_texts: list[str] = field(default_factory=list)

    async def infer_action(
        self, *, text: str, persona: str = "mom", forced_intent: str | None = None,
    ) -> AgentResult:
        self.received_texts.append(text)
        if not self.intent_results:
            raise AssertionError("FakeAgent: пустая очередь intent_results")
        return self.intent_results.pop(0)

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):
        self.semantic_calls += 1
        return list(self.semantic_hits)


def _board(cards: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """cards: [(card_id, name, list_id)] — без чеклистов."""
    return [
        {
            "id": c_id,
            "name": c_name,
            "idList": list_id,
            "closed": False,
            "checklists": [],
        }
        for c_id, c_name, list_id in cards
    ]


# --------------------------------------------------------------------------- tests


def test_fuzzy_strong_match_skips_llm_search():
    """Когда fuzzy уверенно нашёл карточку — LLM не дёргается."""
    board = _board([
        ("car-id", "Обслуживание машины", "DOING"),
        ("milk-id", "Купить молоко", "DOING"),
    ])
    agent = FakeAgent(
        intent_results=[
            AgentResult(
                action=AgentAction(
                    action_type="complete_card_by_text", match_text="обслуживание машины",
                ),
                response_text="ok",
            )
        ],
    )
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent, trello, repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "обслуживание машины"))

    assert any(c[0] == "move_card" for c in trello.calls)
    assert agent.semantic_calls == 0  # LLM-поисковик не вызвался
    assert "обслуживание машины" in out.text.lower()


def test_fuzzy_weak_falls_back_to_semantic_match():
    """Fuzzy слабый, но семантика нашла. Закрываем карточку."""
    board = _board([
        ("car-id", "Обслуживание машины", "DOING"),
        ("foo-id", "Закупка продуктов", "DOING"),
    ])
    # Жаргон "тачка" слабо матчится по название fuzzy, но LLM может «склеить» с обслуживанием машины.
    agent = FakeAgent(
        intent_results=[
            AgentResult(
                action=AgentAction(
                    action_type="complete_card_by_text", match_text="тачка готова",
                ),
                response_text="ok",
            )
        ],
        semantic_hits=[
            CardSearchHit(
                card_id="car-id",
                card_name="Обслуживание машины",
                score=0.9,
                reason="тачка = машина",
                incomplete_items=(),
                total_items=0,
                list_id=None,
            )
        ],
    )
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent, trello, repo)
    profile = FakeProfile()

    _run(orch.process_text(profile, "тачка готова, можно закрывать"))

    assert agent.semantic_calls == 1  # семантический агент сработал
    move_calls = [c for c in trello.calls if c[0] == "move_card"]
    assert len(move_calls) == 1
    assert move_calls[0][1]["card_id"] == "car-id"


def test_semantic_with_two_close_candidates_asks_to_choose():
    """Два кандидата с близким score → вопрос пользователю."""
    board = _board([
        ("car-id", "Обслуживание машины", "DOING"),
        ("buy-id", "Купить машину", "DOING"),
    ])
    agent = FakeAgent(
        intent_results=[
            AgentResult(
                action=AgentAction(
                    action_type="complete_card_by_text", match_text="закрыть про машину",
                ),
                response_text="ok",
            )
        ],
        semantic_hits=[
            CardSearchHit(
                card_id="car-id", card_name="Обслуживание машины",
                score=0.7, reason="машина", incomplete_items=(), total_items=0, list_id=None,
            ),
            CardSearchHit(
                card_id="buy-id", card_name="Купить машину",
                score=0.65, reason="машина", incomplete_items=(), total_items=0, list_id=None,
            ),
        ],
    )
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent, trello, repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "закрой что-то про машину"))

    assert all(c[0] != "move_card" for c in trello.calls)
    assert "выберите номер" in out.text.lower() or "несколько подходящих" in out.text.lower()
    assert profile.telegram_user_id in repo.store


def test_invalid_ids_from_llm_are_filtered():
    """Если LLM вернул несуществующие id — оркестратор их игнорирует."""
    board = _board([("real-id", "Реальная карточка", "DOING")])
    agent = FakeAgent(
        intent_results=[
            AgentResult(
                action=AgentAction(
                    action_type="complete_card_by_text", match_text="что-то странное",
                ),
                response_text="ok",
            )
        ],
        semantic_hits=[
            # Этого id нет в board → должен быть отфильтрован при merge
            CardSearchHit(
                card_id="ghost-id", card_name="Призрак",
                score=0.95, reason="?", incomplete_items=(), total_items=0, list_id=None,
            ),
        ],
    )
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    orch = TaskOrchestrator(agent, trello, repo)
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "что-то странное"))

    assert all(c[0] != "move_card" for c in trello.calls)
    assert "не нашёл" in out.text.lower() or "уточните" in out.text.lower()


def test_merge_results_uses_full_weight_when_one_source_empty():
    """Если semantic пустой, fuzzy не отсекается из-за половинного веса."""
    cards = [
        CardRef(card_id="a", card_name="A", list_id="L", short_url=None,
                incomplete_items=(), total_items=0),
        CardRef(card_id="b", card_name="B", list_id="L", short_url=None,
                incomplete_items=(), total_items=0),
    ]
    fuzzy = [(cards[0], 0.6), (cards[1], 0.55)]
    merged = merge_search_results(fuzzy, [], cards, min_score=0.3)
    assert len(merged) == 2
    assert merged[0].card_id == "a"
    assert merged[0].score >= 0.6 - 1e-6
