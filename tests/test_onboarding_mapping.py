"""Распознавание колонок Trello при /link (в т.ч. русские названия, без ложного «done»)."""

from __future__ import annotations

import pytest

from app.services.onboarding import _match_done, _match_inbox


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Backlog", True),
        ("Бэклог", True),
        ("Product Backlog", True),
        ("To Do", True),
        ("Входящие", True),
        ("Done", False),
        ("Завершённые", False),
    ],
)
def test_match_inbox(name, expected):
    assert _match_inbox(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Done", True),
        ("Completed", True),
        ("Завершённые", True),
        ("Выполнено", True),
        ("Готово", True),
        ("Incomplete", False),
        ("Не завершено", False),
        ("Backlog", False),
        # Раньше «release» в названии + «backlog» давали ложный done и скрывали бэклог в /cards.
        ("Product Release Backlog", False),
        ("Sprint Release Backlog", False),
        ("Release / Done", True),
    ],
)
def test_match_done(name, expected):
    assert _match_done(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Беклог", True),
        ("беклог без ё", True),
    ],
)
def test_match_inbox_beklog_variant(name, expected):
    assert _match_inbox(name) is expected
