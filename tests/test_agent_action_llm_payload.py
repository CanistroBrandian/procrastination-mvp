from app.schemas.actions import AgentAction


def test_metadata_null_and_checklist_list():
    raw = {
        "action_type": "create_checklist_item",
        "card_id": "abc",
        "checklist_item": ["Купить билеты", "купить попкорн"],
        "metadata": None,
        "response_text": "ok",
    }
    a = AgentAction.model_validate(raw)
    assert a.metadata == {}
    assert "Купить билеты" in (a.checklist_item or "")
    assert "попкорн" in (a.checklist_item or "")
