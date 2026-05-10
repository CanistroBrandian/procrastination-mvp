"""Фильтрация карточек в архивных колонках Trello."""

from __future__ import annotations

from app.services.trello_board_filters import (
    archived_list_ids_from_trello_lists,
    drop_cards_in_archived_lists,
)


def test_archived_list_ids():
    lists = [
        {"id": "a", "closed": False},
        {"id": "b", "closed": True},
        {"id": "c", "name": "x"},
    ]
    assert archived_list_ids_from_trello_lists(lists) == {"b"}


def test_drop_cards_in_archived_lists():
    lists = [{"id": "arch", "closed": True}, {"id": "open", "closed": False}]
    cards = [
        {"id": "1", "idList": "open"},
        {"id": "2", "idList": "arch"},
    ]
    out = drop_cards_in_archived_lists(cards, lists)
    assert len(out) == 1 and out[0]["id"] == "1"
