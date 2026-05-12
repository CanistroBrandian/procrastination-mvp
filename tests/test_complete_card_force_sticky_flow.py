from __future__ import annotations

import json

from app.schemas.actions import AgentAction
from app.schemas.actions import AgentResult
from app.services.orchestrator import TaskOrchestrator
from tests.test_complete_card_flow import FakeAgent
from tests.test_complete_card_flow import FakeClarificationRepo
from tests.test_complete_card_flow import FakePending
from tests.test_complete_card_flow import FakeProfile
from tests.test_complete_card_flow import FakeTrello
from tests.test_complete_card_flow import _board_with
from tests.test_complete_card_flow import _run


def _set_complete_card_force_pending(repo: FakeClarificationRepo, profile: FakeProfile) -> None:
    question = (
        "В карточке «Домашние дела: мусор, газон и покупки» есть невыполненные пункты. "
        "Всё равно перевести в «Завершённые»? Ответьте «да» или «нет»."
    )
    pending_action = AgentAction(
        action_type="ask_for_clarification",
        question=question,
        metadata={
            "after_clarification": "complete_card_force",
            "card_id": "card-home",
            "card_name": "Домашние дела: мусор, газон и покупки",
            "remaining_items": [
                "Убрать мусор",
                "Привести в порядок газон",
                "Купить грунт",
                "Посадить дерево",
            ],
        },
    )
    repo.store[profile.telegram_user_id] = FakePending(
        pending_action.model_dump_json(),
        question=question,
    )


def test_complete_card_force_no_then_all_except_keeps_context():
    board = _board_with(
        [
            (
                "card-home",
                "Домашние дела: мусор, газон и покупки",
                "DOING",
                [
                    ("ci-1", "Убрать мусор", "incomplete"),
                    ("ci-2", "Привести в порядок газон", "incomplete"),
                    ("ci-3", "Купить грунт", "incomplete"),
                    ("ci-4", "Посадить дерево", "incomplete"),
                ],
            ),
        ]
    )
    agent = FakeAgent([])
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    profile = FakeProfile()
    _set_complete_card_force_pending(repo, profile)
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)

    out = _run(orch.process_text(profile, "Нет, давай закроем все, кроме купить грунт."))

    update_calls = [c for c in trello.calls if c[0] == "update_check_item_on_card"]
    updated_ids = {c[1]["check_item_id"] for c in update_calls}
    assert len(update_calls) == 3
    assert updated_ids == {"ci-1", "ci-2", "ci-4"}
    assert all(c[0] != "move_card" for c in trello.calls)
    assert profile.telegram_user_id in repo.store
    assert "не перевожу" in out.text.lower() or "остались" in out.text.lower()


def test_complete_card_force_new_command_asks_switch_confirmation():
    board = _board_with(
        [
            (
                "card-home",
                "Домашние дела: мусор, газон и покупки",
                "DOING",
                [("ci-1", "Купить грунт", "incomplete")],
            ),
        ]
    )
    agent = FakeAgent(
        [
            AgentResult(
                action=AgentAction(action_type="none"),
                response_text="Новая команда принята.",
            ),
        ]
    )
    trello = FakeTrello(board_payload=board)
    repo = FakeClarificationRepo()
    profile = FakeProfile()
    _set_complete_card_force_pending(repo, profile)
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)

    first = _run(orch.process_text(profile, "Поставь новую задачу купить хлеб"))
    assert "переключиться" in first.text.lower() and "продолжить" in first.text.lower()
    pending = repo.store.get(profile.telegram_user_id)
    assert pending is not None
    payload = json.loads(pending.draft_action_json)
    assert payload.get("metadata", {}).get("after_clarification") == "confirm_flow_switch"
    assert agent.received_texts == []

    second = _run(orch.process_text(profile, "переключиться"))
    assert second.text == "Новая команда принята."
    assert agent.received_texts == ["Поставь новую задачу купить хлеб"]
