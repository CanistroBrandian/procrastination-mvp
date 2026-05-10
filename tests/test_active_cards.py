"""Тесты парсера /cards и фильтрации по дедлайну."""

from __future__ import annotations

from datetime import date

import pytest

from app.services.active_cards import (
    CardsDisplayFilter,
    _card_matches_filter,
    parse_cards_command_arguments,
)


def test_parse_cards_variants():
    assert parse_cards_command_arguments("/cards") == CardsDisplayFilter(kind="all")
    assert parse_cards_command_arguments("/cards ") == CardsDisplayFilter(kind="all")
    assert parse_cards_command_arguments("/cards 14") == CardsDisplayFilter(kind="due_within", days=14)
    assert parse_cards_command_arguments("/cards неделя") == CardsDisplayFilter(kind="week", days=7)
    assert parse_cards_command_arguments("/cards сегодня") == CardsDisplayFilter(kind="today")
    assert parse_cards_command_arguments("/cards завтра") == CardsDisplayFilter(kind="tomorrow")
    assert parse_cards_command_arguments("/cards просрочка") == CardsDisplayFilter(kind="overdue")
    assert parse_cards_command_arguments("/cards 0") is None
    assert parse_cards_command_arguments("/cards foo") is None


@pytest.mark.parametrize(
    "flt,today,due_iso,due_complete,expected",
    [
        (CardsDisplayFilter(kind="all"), date(2026, 5, 10), None, False, True),
        (
            CardsDisplayFilter(kind="today"),
            date(2026, 5, 10),
            "2026-05-10T12:00:00.000Z",
            False,
            True,
        ),
        (
            CardsDisplayFilter(kind="tomorrow"),
            date(2026, 5, 10),
            "2026-05-11T09:00:00.000Z",
            False,
            True,
        ),
        (
            CardsDisplayFilter(kind="due_within", days=3),
            date(2026, 5, 10),
            "2026-05-12T09:00:00.000Z",
            False,
            True,
        ),
        (
            CardsDisplayFilter(kind="due_within", days=3),
            date(2026, 5, 10),
            "2026-05-14T09:00:00.000Z",
            False,
            False,
        ),
        (
            CardsDisplayFilter(kind="overdue"),
            date(2026, 5, 10),
            "2026-05-07T12:00:00.000Z",
            False,
            True,
        ),
        (
            CardsDisplayFilter(kind="overdue"),
            date(2026, 5, 10),
            "2026-05-10T08:00:00.000Z",
            False,
            False,
        ),
        (
            CardsDisplayFilter(kind="today"),
            date(2026, 5, 10),
            "2026-05-10T12:00:00.000Z",
            True,
            False,
        ),
    ],
)
def test_card_matches_filter(flt, today, due_iso, due_complete, expected):
    card = {"due": due_iso, "dueComplete": due_complete}
    assert _card_matches_filter(card, flt, today=today) is expected
