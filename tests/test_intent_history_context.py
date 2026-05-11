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


class FakeClarificationRepo:
    async def get(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str) -> None:  # noqa: ARG002
        return None

    async def clear(self, telegram_user_id: int) -> None:  # noqa: ARG002
        return None


class FakeTrello:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def create_card(self, list_id, name, desc="", **kw):  # type: ignore[no-untyped-def]
        self.calls.append(("create_card", {"list_id": list_id, "name": name, "desc": desc, **kw}))
        return {"id": "card-1", "name": name}


class ContextAwareAgent:
    def __init__(self, queued: list[AgentResult]) -> None:
        self.queued = list(queued)
        self.received_contexts: list[str | None] = []

    async def infer_action(  # type: ignore[no-untyped-def]
        self,
        *,
        text: str,  # noqa: ARG002
        persona: str = "mom",  # noqa: ARG002
        forced_intent: str | None = None,  # noqa: ARG002
        intent_context: str | None = None,
    ) -> AgentResult:
        self.received_contexts.append(intent_context)
        return self.queued.pop(0)

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):  # noqa: ARG002
        return []


class LegacyAgentNoContextArg:
    def __init__(self, queued: list[AgentResult]) -> None:
        self.queued = list(queued)

    async def infer_action(  # type: ignore[no-untyped-def]
        self,
        *,
        text: str,  # noqa: ARG002
        persona: str = "mom",  # noqa: ARG002
        forced_intent: str | None = None,  # noqa: ARG002
    ) -> AgentResult:
        return self.queued.pop(0)

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):  # noqa: ARG002
        return []


def _run(coro):
    return asyncio.run(coro)


def test_orchestrator_passes_intent_context_to_agent():
    agent = ContextAwareAgent(
        [
            AgentResult(
                action=AgentAction(
                    action_type="create_card",
                    card_name="Купить хлеб",
                    due="2026-05-12T12:00:00+03:00",
                ),
                response_text="Создал.",
            ),
        ],
    )
    trello = FakeTrello()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=FakeClarificationRepo())
    profile = FakeProfile()
    context = "Recent user intent history:\n- intent=update_card; user='перенеси дедлайн'"

    out = _run(orch.process_text(profile, "Поставь задачу купить хлеб", intent_context=context))

    assert out.text == "Создал."
    assert agent.received_contexts == [context]
    assert any(c[0] == "create_card" for c in trello.calls)


def test_orchestrator_keeps_backward_compat_without_intent_context_kwarg():
    agent = LegacyAgentNoContextArg(
        [
            AgentResult(
                action=AgentAction(action_type="none"),
                response_text="Ок.",
            ),
        ],
    )
    trello = FakeTrello()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=FakeClarificationRepo())
    profile = FakeProfile()

    out = _run(orch.process_text(profile, "Привет"))
    assert out.text == "Ок."
