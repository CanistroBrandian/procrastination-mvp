from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.db.models import Persona
from app.schemas.actions import AgentAction, AgentResult
from app.services.orchestrator import TaskOrchestrator


@dataclass
class _Profile:
    telegram_user_id: int = 7
    persona: Persona = Persona.MOM
    trello_board_id: str | None = "B1"
    trello_inbox_list_id: str | None = "L1"
    trello_doing_list_id: str | None = "L2"
    trello_done_list_id: str | None = "L3"


class _FakeClarificationRepo:
    async def get(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def clear(self, telegram_user_id: int):  # noqa: ARG002
        return None

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str):  # noqa: ARG002
        return None


class _FakeTrello:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_board_labels(self, board_id: str):  # noqa: ARG002
        return [{"id": "lbl-1", "name": "Срочно", "color": "red"}]

    async def add_label_to_card(self, card_id: str, label_id: str):
        self.calls.append(("add_label_to_card", {"card_id": card_id, "label_id": label_id}))
        return {"ok": True}

    async def list_board_members(self, board_id: str):  # noqa: ARG002
        return [{"id": "mem-1", "username": "ivan", "fullName": "Иван Петров", "initials": "IP"}]

    async def add_member_to_card(self, card_id: str, member_id: str):
        self.calls.append(("add_member_to_card", {"card_id": card_id, "member_id": member_id}))
        return {"ok": True}


class _AgentLabel:
    async def infer_action(self, **kwargs):  # noqa: ARG002
        return AgentResult(
            action=AgentAction(action_type="add_card_label", card_id="c-1", label_id="Срочно"),
            response_text="ok",
        )


class _AgentMember:
    async def infer_action(self, **kwargs):  # noqa: ARG002
        return AgentResult(
            action=AgentAction(action_type="add_card_member", card_id="c-1", member_id="@ivan"),
            response_text="ok",
        )


def _run(coro):
    return asyncio.run(coro)


def test_add_label_resolves_name_to_id():
    trello = _FakeTrello()
    orch = TaskOrchestrator(
        agent_service=_AgentLabel(),
        trello_client=trello,
        clarification_repo=_FakeClarificationRepo(),
    )
    profile = _Profile()

    _run(orch.process_text(profile, "Поставь метку Срочно"))

    assert ("add_label_to_card", {"card_id": "c-1", "label_id": "lbl-1"}) in trello.calls


def test_add_member_resolves_username_to_id():
    trello = _FakeTrello()
    orch = TaskOrchestrator(
        agent_service=_AgentMember(),
        trello_client=trello,
        clarification_repo=_FakeClarificationRepo(),
    )
    profile = _Profile()

    _run(orch.process_text(profile, "Назначь @ivan"))

    assert ("add_member_to_card", {"card_id": "c-1", "member_id": "mem-1"}) in trello.calls
