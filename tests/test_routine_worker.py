from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from app.db.models import RoutineTemplate, UserProfile
from app.workers import routines as routines_worker
from app.workers.routines import run_routine_generation


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)


class FakeSession:
    def __init__(self, users):
        self.users = users

    async def execute(self, stmt):  # noqa: ARG002
        return _Result(self.users)


class FakeProfileRepo:
    def __init__(self, users):
        self.session = FakeSession(users)


class FakeRoutinesRepo:
    def __init__(self, templates):
        self.templates = templates

    async def list_active(self):
        return self.templates

    async def mark_generated(self, template_id: int, date_yyyy_mm_dd: str):
        for t in self.templates:
            if t.id == template_id:
                t.last_generated_date = date_yyyy_mm_dd

    async def mark_generated_if_not_date(self, template_id: int, date_yyyy_mm_dd: str) -> bool:
        for t in self.templates:
            if t.id != template_id:
                continue
            if t.last_generated_date == date_yyyy_mm_dd:
                return False
            t.last_generated_date = date_yyyy_mm_dd
            return True
        return False


class FakeTaskEvents:
    def __init__(self):
        self.events = []

    async def append(self, *, telegram_user_id: int, card_id: str | None, event_type: str, category_key: str | None = None):
        self.events.append((telegram_user_id, card_id, event_type, category_key))


class FakeTrello:
    def __init__(self):
        self.api_key = "k"
        self.api_token = "t"
        self.created_cards = []

    async def create_card(self, list_id: str, name: str, desc: str = "", **kwargs):  # noqa: ARG002
        card_id = f"c-{len(self.created_cards) + 1}"
        self.created_cards.append((list_id, name))
        return {"id": card_id, "name": name}

    async def add_checklist(self, card_id: str, name: str):  # noqa: ARG002
        return {"id": "cl-1"}

    async def add_check_item(self, checklist_id: str, item_name: str):  # noqa: ARG002
        return {"id": f"it-{item_name}"}


def _run(coro):
    return asyncio.run(coro)


def test_routine_worker_idempotent_per_day():
    user = UserProfile(telegram_user_id=11, trello_board_id="B1", trello_inbox_list_id="L1", timezone="Europe/Moscow")
    template = RoutineTemplate(
        id=1,
        telegram_user_id=11,
        name="English daily",
        category_key=None,
        duration_min=60,
        checklist_template_json='["lesson"]',
        schedule_cron="* * * * *",
        is_active=1,
        last_generated_date=None,
    )
    profile_repo = FakeProfileRepo([user])
    routines_repo = FakeRoutinesRepo([template])
    events = FakeTaskEvents()
    trello = FakeTrello()

    _run(run_routine_generation(profile_repo, routines_repo, events, trello))
    first_count = len(trello.created_cards)
    _run(run_routine_generation(profile_repo, routines_repo, events, trello))
    second_count = len(trello.created_cards)

    assert first_count == 1
    assert second_count == 1
    assert any(ev[2] == "routine_generated" for ev in events.events)


def test_routine_worker_matches_user_cron_in_tick_window(monkeypatch):
    user = UserProfile(telegram_user_id=12, trello_board_id="B2", trello_inbox_list_id="L1", timezone="Europe/Moscow")
    template = RoutineTemplate(
        id=2,
        telegram_user_id=12,
        name="Daily standup note",
        category_key=None,
        duration_min=15,
        checklist_template_json=None,
        schedule_cron="7 * * * *",
        is_active=1,
        last_generated_date=None,
    )
    profile_repo = FakeProfileRepo([user])
    routines_repo = FakeRoutinesRepo([template])
    events = FakeTaskEvents()
    trello = FakeTrello()

    monkeypatch.setattr(
        routines_worker,
        "now_in_user_tz",
        lambda profile, fallback_tz="Europe/Moscow": datetime(2026, 5, 14, 13, 10),
    )

    _run(run_routine_generation(profile_repo, routines_repo, events, trello, tick_minutes=10))

    assert len(trello.created_cards) == 1


def test_routine_worker_fallbacks_invalid_template_cron(monkeypatch):
    user = UserProfile(
        telegram_user_id=13,
        trello_board_id="B3",
        trello_inbox_list_id="L1",
        timezone="Europe/Moscow",
        routine_cron="* * * * *",
    )
    template = RoutineTemplate(
        id=3,
        telegram_user_id=13,
        name="Invalid cron template",
        category_key=None,
        duration_min=10,
        checklist_template_json=None,
        schedule_cron="not-a-cron",
        is_active=1,
        last_generated_date=None,
    )
    profile_repo = FakeProfileRepo([user])
    routines_repo = FakeRoutinesRepo([template])
    events = FakeTaskEvents()
    trello = FakeTrello()

    monkeypatch.setattr(
        routines_worker,
        "now_in_user_tz",
        lambda profile, fallback_tz="Europe/Moscow": datetime(2026, 5, 14, 13, 10),
    )

    _run(run_routine_generation(profile_repo, routines_repo, events, trello, tick_minutes=10))

    assert len(trello.created_cards) == 1
