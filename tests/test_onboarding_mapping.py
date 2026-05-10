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
    ],
)
def test_match_done(name, expected):
    assert _match_done(name) is expected
