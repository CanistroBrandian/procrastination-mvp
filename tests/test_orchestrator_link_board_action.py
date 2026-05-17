from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.db.models import Persona
from app.schemas.actions import AgentAction, AgentResult
from app.services.orchestrator import TaskOrchestrator


@dataclass
class _Profile:
    telegram_user_id: int = 1
    persona: Persona = Persona.MOM
    trello_board_id: str | None = None
    trello_inbox_list_id: str | None = None
    trello_doing_list_id: str | None = None
    trello_done_list_id: str | None = None


class _FakeAgent:
    async def infer_action(self, **kwargs):  # noqa: ARG002
        return AgentResult(
            action=AgentAction(
                action_type="link_board",
                metadata={"board_url": "https://trello.com/b/GMK1lbRp/work"},
            ),
            response_text="ok",
        )


class _FakeClarificationRepo:
    async def get(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def clear(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str):  # noqa: ARG002
        return None


class _FakeTrello:
    async def list_lists(self, board_id: str):  # noqa: ARG002
        return [
            {"id": "inbox", "name": "Inbox", "closed": False},
            {"id": "doing", "name": "Doing", "closed": False},
            {"id": "done", "name": "Done", "closed": False},
        ]


def _run(coro):
    return asyncio.run(coro)


def test_orchestrator_handles_link_board_action():
    profile = _Profile()
    orch = TaskOrchestrator(
        agent_service=_FakeAgent(),
        trello_client=_FakeTrello(),
        clarification_repo=_FakeClarificationRepo(),
    )

    out = _run(orch.process_text(profile, "https://trello.com/b/GMK1lbRp/work"))

    assert profile.trello_board_id == "GMK1lbRp"
    assert "Доска подключена" in out.text
