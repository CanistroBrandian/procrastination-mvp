"""Интеграция датасета NeuroNudge с AgentService: мок LLM (`_call_llm_json`).

- Роутер не вызывается по сети: подставляем JSON шага `router` с intent из сценария.
- Rule-based обходим (`quick_classify_intent → None`), чтобы всегда проверять ветку «как LLM».
- Экстрактор подставляет минимально валидные поля под intent.

Так мы гарантируем, что все 100 разговорных реплик проходят через `infer_action`
без падений валидации Pydantic — регрессии при смене схем будут видны сразу.

Живой роутер без мока: см. `test_neuronudge_live_router_sample` (отключён по умолчанию).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai import AsyncOpenAI

from app.services.agent import AgentService
from app.core.config import Settings
from app.core.llm_client import build_async_openai_client
from tests.neuronudge_dataset import NeuronudgeScenario, load_neuronudge_scenarios, scenario_id


def _minimal_extractor_payload(intent: str) -> dict[str, Any]:
    """Минимальный ответ «экстрактора», достаточный для AgentAction."""
    ok = "Ок."
    if intent == "create_card":
        return {
            "card_name": "Задача из датасета",
            "due": None,
            "start": None,
            "position": None,
            "checklist_name": None,
            "checklist_item": None,
            "card_description": None,
            "response_text": ok,
        }
    if intent == "complete_card_by_text":
        return {"match_text": "тестовая карточка", "response_text": ok}
    if intent == "complete_task_from_text":
        return {"match_text": "пункт чеклиста", "response_text": ok}
    if intent == "add_checklist_items":
        return {
            "card_id": None,
            "checklist_name": "Шаги",
            "checklist_item": ["пункт 1"],
            "response_text": ok,
        }
    if intent == "update_card":
        return {"card_id": None, "card_name": None, "due": None, "response_text": ok}
    if intent == "delete_card":
        return {"card_name": "тест", "response_text": ok}
    if intent == "move_card":
        return {"card_name": None, "list_name": "doing", "response_text": ok}
    if intent == "link_board":
        return {"metadata": {"board_url": "https://trello.com/b/abc123/board-name"}, "response_text": ok}
    if intent == "set_persona":
        return {"metadata": {"persona": "mom"}, "response_text": ok}
    if intent in ("add_comment",):
        return {"comment_text": "коммент", "response_text": ok}
    if intent in ("add_card_label", "remove_card_label", "add_card_member", "remove_card_member"):
        return {"response_text": ok}
    if intent == "attach_file":
        return {"file_url": "https://example.com/f", "response_text": ok}
    if intent == "update_checklist_item":
        return {"response_text": ok}
    raise NotImplementedError(f"Добавьте minimal payload для intent={intent!r} в тесте")


def _expected_action_type(intent: str) -> str:
    if intent == "add_checklist_items":
        return "create_checklist_item"
    return intent


async def _infer_with_mocks(scenario: NeuronudgeScenario):
    agent = AgentService(client=MagicMock(spec=AsyncOpenAI), model="mock-model")

    async def fake_llm(_system: str, user: str, *, step: str) -> dict[str, Any]:
        exp = scenario.expected_router_intent
        if step == "router":
            return {
                "intent": exp,
                "reasoning": "mock_router",
                "response_text": "Принято.",
            }
        if not step.startswith("extractor:"):
            raise AssertionError(f"неожиданный step={step!r}")
        step_intent = step.removeprefix("extractor:")

        assert step_intent == exp, f"роутер выбрал {exp}, а экстрактор ждёт {step_intent}"
        return _minimal_extractor_payload(exp)

    with patch.object(agent, "_call_llm_json", side_effect=fake_llm):
        with patch("app.services.agent.quick_classify_intent", return_value=None):
            return await agent.infer_action(scenario.user_text, persona="mom")


@pytest.mark.parametrize(
    "scenario",
    load_neuronudge_scenarios(),
    ids=scenario_id,
)
def test_infer_action_full_pipeline_mocked_llm(scenario: NeuronudgeScenario):
    """Полный infer_action: мок роутера + экстрактора; реплика пользователя из датасета."""

    async def run():
        return await _infer_with_mocks(scenario)

    result = asyncio.run(run())
    exp = scenario.expected_router_intent

    if exp == "chitchat":
        assert result.action.action_type == "none"
        return

    assert result.action.action_type == _expected_action_type(exp)


@pytest.mark.skipif(
    os.getenv("RUN_NEURONUDGE_LIVE_ROUTER") != "1",
    reason="Установите RUN_NEURONUDGE_LIVE_ROUTER=1 и ключ OPENROUTER_API_KEY или OPENAI_API_KEY в окружении",
)
def test_neuronudge_live_router_sample():
    """Живой вызов `_classify_intent` на первых N репликах — для отладки промпта вручную.

    Не включайте в CI без ключей. Ожидается intent create_card (или NEURONUDGE_INTENT_OVERRIDES).
    """
    settings = Settings()
    if not (settings.openrouter_api_key or "").strip() and not (settings.openai_api_key or "").strip():
        pytest.skip("Нет OPENROUTER_API_KEY / OPENAI_API_KEY")

    client = build_async_openai_client(settings)
    agent = AgentService(client=client, model=settings.chat_model)
    scenarios = load_neuronudge_scenarios()[:15]

    async def run_batch():
        out: list[tuple[NeuronudgeScenario, str]] = []
        for s in scenarios:
            with patch("app.services.agent.quick_classify_intent", return_value=None):
                d = await agent._classify_intent(s.user_text, persona="mom")
            out.append((s, d.intent))
        return out

    pairs = asyncio.run(run_batch())
    mismatches = []
    for sc, got in pairs:
        exp = sc.expected_router_intent
        if got != exp:
            mismatches.append((sc.row_index, sc.task_title[:48], exp, got))

    assert not mismatches, "Роутер расходится с ожиданием датасета:\n" + "\n".join(
        f"  #{i} {t!r}: ожидали {e}, получили {g}" for i, t, e, g in mismatches
    )
