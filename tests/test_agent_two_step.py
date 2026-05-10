"""Двухшаговая агентная система: роутер → экстрактор.

Проверяем что:
1) Роутер вызывается с компактным промптом и его intent корректно определяет тип действия.
2) Экстрактор вызывается с узким промптом под выбранный intent.
3) Если роутер вернул `chitchat` — экстрактор не дёргается, возвращается action_type=none.
4) `forced_intent` минует роутер.
5) Известные «проблемные» фразы пользователя не попадают в complete_task_from_text по ошибке.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.schemas.actions import AgentResult  # noqa: F401  (используется в типах в будущем)
from app.services.agent import AgentService


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- fakes


@dataclass
class FakeChoice:
    message: Any


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]


class FakeChatCompletions:
    """Возвращает запрограммированные JSON-ответы по очереди."""

    def __init__(self, queued: list[dict[str, Any]]):
        self.queued = list(queued)
        self.captured: list[dict[str, Any]] = []

    async def create(self, *, model, temperature, messages, response_format):  # type: ignore[no-untyped-def]
        self.captured.append(
            {
                "model": model,
                "system": messages[0]["content"],
                "user": messages[1]["content"],
            }
        )
        if not self.queued:
            raise AssertionError("FakeChatCompletions: пустая очередь")
        payload = self.queued.pop(0)
        return FakeCompletion(choices=[FakeChoice(message=FakeMessage(content=json.dumps(payload, ensure_ascii=False)))])


class FakeChat:
    def __init__(self, completions: FakeChatCompletions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, queued: list[dict[str, Any]]):
        self.completions = FakeChatCompletions(queued)
        self.chat = FakeChat(self.completions)


# --------------------------------------------------------------------------- tests


def test_rule_based_skips_llm_router_for_future_tense_homework():
    """«поставим задачу...» — rule-based перехватывает, LLM-роутер не дёргается."""
    client = FakeOpenAI(
        queued=[
            {  # сразу extractor, без router
                "action_type": "create_card",
                "card_name": "Делать уроки",
                "due": "2026-05-11T09:00:00.000Z",
                "checklist_item": [],
                "response_text": "Окей.",
            },
        ]
    )
    svc = AgentService(client=client, model="x")  # type: ignore[arg-type]
    result = _run(svc.infer_action("Слушай, давай на завтра поставим задачу делать уроки", persona="mom"))

    assert result.action.action_type == "create_card"
    assert result.action.card_name == "Делать уроки"
    # Только один вызов — extractor. Router был rule-based.
    assert len(client.completions.captured) == 1
    assert "create_card" in client.completions.captured[0]["system"].lower()


def test_rule_based_skips_llm_router_for_planting_tree():
    """«зададим задачу посадить дерево» — rule-based перехватывает."""
    client = FakeOpenAI(
        queued=[
            {
                "action_type": "create_card",
                "card_name": "Посадить дерево",
                "due": "2026-05-11T09:00:00.000Z",
                "checklist_name": "Шаги",
                "checklist_item": ["Купить грунт", "Купить лопату", "Забрать сына"],
                "response_text": "Готово.",
            },
        ]
    )
    svc = AgentService(client=client, model="x")  # type: ignore[arg-type]
    text = (
        "короче давай завтра зададим задачу посадить дерево и там для этого нам нужно "
        "купить грунт и купить лопату а еще нужно будет взять и забрать сына"
    )
    result = _run(svc.infer_action(text, persona="mom"))

    assert result.action.action_type == "create_card"
    assert "посадить дерево" in (result.action.card_name or "").lower()
    assert "Купить грунт" in (result.action.checklist_item or "")
    assert len(client.completions.captured) == 1


def test_rule_based_skips_llm_router_for_finished_card_phrase():
    """«машину обслужили, можно закрывать эту задачу» — rule-based → complete_card."""
    client = FakeOpenAI(
        queued=[
            {
                "action_type": "complete_card_by_text",
                "match_text": "обслуживание машины",
                "response_text": "Перевожу карточку.",
            },
        ]
    )
    svc = AgentService(client=client, model="x")  # type: ignore[arg-type]
    result = _run(svc.infer_action(
        "так вы машину мы обслужили в принципе можно закрывать эту задачу",
        persona="mom",
    ))

    assert result.action.action_type == "complete_card_by_text"
    assert result.action.match_text == "обслуживание машины"
    # LLM-роутер не дёргался: rule-based сработал.
    assert len(client.completions.captured) == 1
    assert "complete_card_by_text" in client.completions.captured[0]["system"].lower()


def test_router_picks_complete_task_for_past_tense_subtask():
    client = FakeOpenAI(
        queued=[
            {
                "intent": "complete_task_from_text",
                "reasoning": "прошедшее время о подзадаче",
                "response_text": "Отмечаю.",
            },
            {
                "action_type": "complete_task_from_text",
                "match_text": "помыл лобовое стекло",
                "response_text": "Отмечаю.",
            },
        ]
    )
    svc = AgentService(client=client, model="x")  # type: ignore[arg-type]
    result = _run(svc.infer_action("Помыл лобовое стекло", persona="mom"))

    assert result.action.action_type == "complete_task_from_text"
    assert result.action.match_text == "помыл лобовое стекло"


def test_chitchat_skips_extractor():
    client = FakeOpenAI(
        queued=[
            {
                "intent": "chitchat",
                "reasoning": "приветствие",
                "response_text": "Привет!",
            },
        ]
    )
    svc = AgentService(client=client, model="x")  # type: ignore[arg-type]
    result = _run(svc.infer_action("Привет, как дела?", persona="mom"))

    assert result.action.action_type == "none"
    assert result.response_text == "Привет!"
    # Только один вызов LLM — экстрактор не дёргался.
    assert len(client.completions.captured) == 1


def test_forced_intent_skips_router():
    client = FakeOpenAI(
        queued=[
            {
                "action_type": "create_card",
                "card_name": "Купить молоко",
                "due": "2026-05-12T18:00:00.000Z",
                "response_text": "Записал.",
            },
        ]
    )
    svc = AgentService(client=client, model="x")  # type: ignore[arg-type]
    result = _run(svc.infer_action("Купить молоко завтра в 18:00", persona="mom", forced_intent="create_card"))

    assert result.action.action_type == "create_card"
    # Только экстрактор — никаких роутеров.
    assert len(client.completions.captured) == 1
    assert "create_card" in client.completions.captured[0]["system"].lower()


def test_extractor_response_normalizes_action_type_to_intent():
    """Если экстрактор перепутал action_type — оркестратор всё равно получит правильный."""
    client = FakeOpenAI(
        queued=[
            {
                "intent": "complete_card_by_text",
                "reasoning": "вся карточка",
                "response_text": "Закрываю.",
            },
            {
                # Симулируем что extractor по ошибке поставил complete_task_from_text.
                "action_type": "complete_task_from_text",
                "match_text": "обслуживание машины",
                "response_text": "Закрываю.",
            },
        ]
    )
    svc = AgentService(client=client, model="x")  # type: ignore[arg-type]
    result = _run(svc.infer_action("закрой обслуживание машины", persona="mom"))

    # Оркестратор увидит правильный intent, потому что мы его принудительно проставляем.
    assert result.action.action_type == "complete_card_by_text"
    assert result.action.match_text == "обслуживание машины"


def test_router_falls_back_to_chitchat_on_unknown_intent():
    client = FakeOpenAI(
        queued=[
            {
                "intent": "do_a_backflip",  # невалидный intent
                "reasoning": "?",
                "response_text": "ok",
            },
        ]
    )
    svc = AgentService(client=client, model="x")  # type: ignore[arg-type]
    result = _run(svc.infer_action("...", persona="mom"))

    assert result.action.action_type == "none"
