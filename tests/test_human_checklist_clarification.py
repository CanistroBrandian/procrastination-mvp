from __future__ import annotations

from app.schemas.actions import AgentAction
from app.services.orchestrator import TaskOrchestrator
from tests.test_complete_card_flow import FakeAgent
from tests.test_complete_card_flow import FakeClarificationRepo
from tests.test_complete_card_flow import FakePending
from tests.test_complete_card_flow import FakeProfile
from tests.test_complete_card_flow import FakeTrello
from tests.test_complete_card_flow import _board_with
from tests.test_complete_card_flow import _run


def _orch_with_pending(board_payload, question: str):
    agent = FakeAgent([])
    trello = FakeTrello(board_payload=board_payload)
    repo = FakeClarificationRepo()
    profile = FakeProfile()
    orch = TaskOrchestrator(agent_service=agent, trello_client=trello, clarification_repo=repo)

    pending_action = AgentAction(
        action_type="ask_for_clarification",
        question=question,
        metadata={},
    )
    repo.store[profile.telegram_user_id] = FakePending(
        pending_action.model_dump_json(),
        question=question,
    )
    return orch, trello, repo, profile, agent


def test_pending_human_clarification_all_except_item_name():
    board = _board_with(
        [
            (
                "card-shop",
                "Сходить в магазин",
                "DOING",
                [
                    ("ci-1", "Купить муку", "incomplete"),
                    ("ci-2", "Купить лук", "incomplete"),
                    ("ci-3", "Купить чеснок", "incomplete"),
                    ("ci-4", "Купить молоко", "incomplete"),
                ],
            ),
        ]
    )
    question = "Ты хочешь закрыть всю карточку «Сходить в магазин» или отметить выполненным какой-то пункт внутри неё?"
    orch, trello, _repo, profile, agent = _orch_with_pending(board, question)

    out = _run(orch.process_text(profile, "Отметь все пункты, кроме «Купить чеснок»."))

    update_calls = [c for c in trello.calls if c[0] == "update_check_item_on_card"]
    updated_ids = {c[1]["check_item_id"] for c in update_calls}
    assert len(update_calls) == 3
    assert updated_ids == {"ci-1", "ci-2", "ci-4"}
    assert agent.received_texts == []
    assert "3" in out.text


def test_pending_human_clarification_nothing_done_except_first():
    board = _board_with(
        [
            (
                "card-shop",
                "Сходить в магазин",
                "DOING",
                [
                    ("ci-1", "Купить муку", "incomplete"),
                    ("ci-2", "Купить лук", "incomplete"),
                    ("ci-3", "Купить чеснок", "incomplete"),
                ],
            ),
        ]
    )
    question = "Что именно отметить в карточке «Сходить в магазин»?"
    orch, trello, _repo, profile, agent = _orch_with_pending(board, question)

    out = _run(orch.process_text(profile, "Ничего не сделано кроме первого."))

    update_calls = [c for c in trello.calls if c[0] == "update_check_item_on_card"]
    assert len(update_calls) == 1
    assert update_calls[0][1]["check_item_id"] == "ci-1"
    assert agent.received_texts == []
    assert "1" in out.text


def test_pending_human_clarification_marks_two_indices():
    board = _board_with(
        [
            (
                "card-shop",
                "Сходить в магазин",
                "DOING",
                [
                    ("ci-1", "Купить муку", "incomplete"),
                    ("ci-2", "Купить лук", "incomplete"),
                    ("ci-3", "Купить чеснок", "incomplete"),
                ],
            ),
        ]
    )
    question = "Что отметить в карточке «Сходить в магазин»?"
    orch, trello, _repo, profile, agent = _orch_with_pending(board, question)

    out = _run(orch.process_text(profile, "Закрыть пункт 1 и 2."))

    update_calls = [c for c in trello.calls if c[0] == "update_check_item_on_card"]
    updated_ids = {c[1]["check_item_id"] for c in update_calls}
    assert len(update_calls) == 2
    assert updated_ids == {"ci-1", "ci-2"}
    assert agent.received_texts == []
    assert "2" in out.text
