from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.db.models import Persona
from app.schemas.actions import AgentAction
from app.services.orchestrator import TaskOrchestrator


@dataclass
class FakeProfile:
    telegram_user_id: int = 77
    persona: Persona = Persona.MOM
    trello_board_id: str | None = "BOARD"
    trello_inbox_list_id: str | None = "INBOX"
    trello_doing_list_id: str | None = "DOING"
    trello_done_list_id: str | None = "DONE"


@dataclass
class FakePending:
    draft_action_json: str
    question: str


class FakeClarificationRepo:
    def __init__(self) -> None:
        self.store: dict[int, FakePending] = {}

    async def get(self, telegram_user_id: int):
        return self.store.get(telegram_user_id)

    async def clear(self, telegram_user_id: int) -> None:
        self.store.pop(telegram_user_id, None)

    async def upsert(self, telegram_user_id: int, question: str, draft_action_json: str) -> None:
        self.store[telegram_user_id] = FakePending(draft_action_json=draft_action_json, question=question)


class FakeTrello:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.board_payload = [
            {"id": "c-plan", "name": "План разработки приложения", "idList": "DOING", "closed": False, "checklists": []},
        ]
        self.card_desc: dict[str, str] = {"c-plan": "Исходное описание"}

    async def list_lists(self, board_id: str):  # noqa: ARG002
        self.calls.append(("list_lists", {"board_id": board_id}))
        return [{"id": "DOING", "name": "DOING", "closed": False}]

    async def list_board_cards_with_checklists(self, board_id: str):  # noqa: ARG002
        self.calls.append(("list_board_cards_with_checklists", {"board_id": board_id}))
        return self.board_payload

    async def get_card(self, card_id: str):
        self.calls.append(("get_card", {"card_id": card_id}))
        return {"id": card_id, "name": "План разработки приложения", "desc": self.card_desc.get(card_id, "")}

    async def update_card(self, card_id, name=None, desc=None, **kw):  # type: ignore[no-untyped-def]
        self.calls.append(("update_card", {"card_id": card_id, "name": name, "desc": desc, **kw}))
        if desc is not None:
            self.card_desc[str(card_id)] = str(desc)
        return {"id": card_id}

    async def add_checklist(self, card_id: str, name: str):
        self.calls.append(("add_checklist", {"card_id": card_id, "name": name}))
        return {"id": "cl-links"}

    async def add_check_item(self, checklist_id: str, item_name: str):
        self.calls.append(("add_check_item", {"checklist_id": checklist_id, "item_name": item_name}))
        return {"id": "item-1"}


class FakeAgent:
    async def infer_action(self, *, text: str, persona: str = "mom", forced_intent: str | None = None, intent_context: str | None = None):  # noqa: ARG002
        raise AssertionError(f"infer_action should not be called in attachment pending flow, got text={text!r}")

    async def search_cards_semantic(self, query, cards_payload, persona="mom"):  # noqa: ARG002
        return []


def _run(coro):
    return asyncio.run(coro)


def _set_pending(repo: FakeClarificationRepo, profile: FakeProfile, *, after: str, metadata: dict[str, Any], question: str) -> None:
    action = AgentAction(
        action_type="ask_for_clarification",
        question=question,
        metadata={"after_clarification": after, **metadata},
    )
    repo.store[profile.telegram_user_id] = FakePending(draft_action_json=action.model_dump_json(), question=question)


def test_file_attachment_pending_appends_file_url_to_description():
    repo = FakeClarificationRepo()
    profile = FakeProfile()
    _set_pending(
        repo,
        profile,
        after="attach_asset_pick_card",
        metadata={"asset_kind": "file", "asset_urls": ["https://files.telegram.example/doc1"]},
        question="К какой задаче прикрепить файл?",
    )
    trello = FakeTrello()
    orch = TaskOrchestrator(agent_service=FakeAgent(), trello_client=trello, clarification_repo=repo)

    out = _run(orch.process_text(profile, "План разработки приложения"))

    update_calls = [c for c in trello.calls if c[0] == "update_card"]
    assert len(update_calls) == 1
    assert update_calls[0][1]["card_id"] == "c-plan"
    assert "Файл: https://files.telegram.example/doc1" in (update_calls[0][1]["desc"] or "")
    assert "добавил файл в описание" in out.text.lower()


def test_link_attachment_pending_moves_to_mode_selection():
    repo = FakeClarificationRepo()
    profile = FakeProfile()
    _set_pending(
        repo,
        profile,
        after="attach_asset_pick_card",
        metadata={"asset_kind": "link", "asset_urls": ["https://example.com/spec"]},
        question="К какой задаче прикрепить ссылку?",
    )
    trello = FakeTrello()
    orch = TaskOrchestrator(agent_service=FakeAgent(), trello_client=trello, clarification_repo=repo)

    out = _run(orch.process_text(profile, "План разработки приложения"))

    saved = repo.store.get(profile.telegram_user_id)
    assert saved is not None
    payload = json.loads(saved.draft_action_json)
    assert payload.get("metadata", {}).get("after_clarification") == "attach_link_choose_mode"
    assert "как прикрепить ссылку" in out.text.lower()


def test_link_mode_checklist_creates_checklist_item():
    repo = FakeClarificationRepo()
    profile = FakeProfile()
    _set_pending(
        repo,
        profile,
        after="attach_link_choose_mode",
        metadata={
            "card_id": "c-plan",
            "card_name": "План разработки приложения",
            "asset_urls": ["https://example.com/spec"],
        },
        question="Как прикрепить ссылку?",
    )
    trello = FakeTrello()
    orch = TaskOrchestrator(agent_service=FakeAgent(), trello_client=trello, clarification_repo=repo)

    out = _run(orch.process_text(profile, "1"))

    checklist_calls = [c for c in trello.calls if c[0] == "add_checklist"]
    item_calls = [c for c in trello.calls if c[0] == "add_check_item"]
    assert len(checklist_calls) == 1
    assert checklist_calls[0][1]["card_id"] == "c-plan"
    assert len(item_calls) == 1
    assert item_calls[0][1]["item_name"] == "https://example.com/spec"
    assert "чеклист" in out.text.lower()


def test_link_mode_description_appends_description():
    repo = FakeClarificationRepo()
    profile = FakeProfile()
    _set_pending(
        repo,
        profile,
        after="attach_link_choose_mode",
        metadata={
            "card_id": "c-plan",
            "card_name": "План разработки приложения",
            "asset_urls": ["https://example.com/spec"],
        },
        question="Как прикрепить ссылку?",
    )
    trello = FakeTrello()
    orch = TaskOrchestrator(agent_service=FakeAgent(), trello_client=trello, clarification_repo=repo)

    out = _run(orch.process_text(profile, "в описание"))

    update_calls = [c for c in trello.calls if c[0] == "update_card"]
    assert len(update_calls) == 1
    assert update_calls[0][1]["card_id"] == "c-plan"
    assert "Ссылка: https://example.com/spec" in (update_calls[0][1]["desc"] or "")
    assert "описание" in out.text.lower()
