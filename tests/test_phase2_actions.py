from app.schemas.actions import AgentAction


def test_phase2_action_types_validate():
    a = AgentAction.model_validate(
        {
            "action_type": "add_comment",
            "card_id": "abc123",
            "comment_text": "Заметка",
        }
    )
    assert a.action_type == "add_comment"

    b = AgentAction.model_validate(
        {
            "action_type": "update_checklist_item",
            "card_id": "c",
            "checklist_id": "cl",
            "check_item_id": "ci",
            "check_item_complete": True,
        }
    )
    assert b.check_item_complete is True
