from app.services.checklist_match import (
    CheckItemRef,
    best_matches,
    confident_unique,
    flatten_incomplete_items,
    score,
)


def _ref(name: str, *, card_id="C1", card_name="Карточка", checklist_id="L1", check_item_id="I1") -> CheckItemRef:
    return CheckItemRef(
        card_id=card_id,
        card_name=card_name,
        list_id=None,
        checklist_id=checklist_id,
        checklist_name="Шаги",
        check_item_id=check_item_id,
        item_name=name,
        state="incomplete",
    )


def test_score_morphology_handled():
    assert score("помыл окна", "Помыть окна") > 0.7
    assert score("пропылесосил", "Пропылесосить") > 0.6
    assert score("сделал отчёт", "Помыть окна") < 0.3


def test_best_matches_filters_low_score():
    items = [_ref("Помыть окна", check_item_id="i1"), _ref("Пропылесосить", check_item_id="i2")]
    matches = best_matches("я помыл окна", items)
    assert matches and matches[0][0].check_item_id == "i1"


def test_confident_unique_when_clear_winner():
    items = [_ref("Помыть окна", check_item_id="i1"), _ref("Купить хлеб", check_item_id="i2")]
    matches = best_matches("помыл окна", items)
    assert confident_unique(matches) is not None
    assert confident_unique(matches).check_item_id == "i1"


def test_confident_unique_none_when_tied():
    a = _ref("Сделать отчёт А", check_item_id="i1")
    b = _ref("Сделать отчёт Б", check_item_id="i2")
    matches = best_matches("сделать отчёт", [a, b])
    assert confident_unique(matches) is None


def test_flatten_skips_complete_and_closed_cards():
    payload = [
        {
            "id": "card-open",
            "name": "Open",
            "idList": "list1",
            "closed": False,
            "checklists": [
                {
                    "id": "cl1",
                    "name": "Шаги",
                    "checkItems": [
                        {"id": "i1", "name": "Помыть окна", "state": "incomplete"},
                        {"id": "i2", "name": "Пропылесосить", "state": "complete"},
                    ],
                }
            ],
        },
        {
            "id": "card-closed",
            "name": "Closed",
            "idList": "list1",
            "closed": True,
            "checklists": [
                {"id": "cl2", "name": "x", "checkItems": [{"id": "ix", "name": "Игнор", "state": "incomplete"}]},
            ],
        },
    ]
    refs = flatten_incomplete_items(payload)
    assert [r.item_name for r in refs] == ["Помыть окна"]
