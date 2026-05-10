from datetime import datetime
from unittest.mock import patch

from zoneinfo import ZoneInfo

from app.services.date_infer import infer_due_iso_from_russian

MSK = ZoneInfo("Europe/Moscow")


def _patch_now(fixed: datetime):
    real_datetime = __import__("datetime").datetime

    def factory(mock):
        mock.timedelta = __import__("datetime").timedelta
        mock.now.side_effect = lambda tz=None: fixed
        mock.side_effect = lambda *a, **k: real_datetime(*a, **k)

    return factory


@patch("app.services.date_infer.datetime")
def test_in_friday_resolves_to_next_friday(mock_dt):
    fixed = datetime(2026, 5, 11, 10, 0, 0, tzinfo=MSK)  # Monday
    _patch_now(fixed)(mock_dt)

    out = infer_due_iso_from_russian("позвонить в пятницу")
    assert out is not None
    assert out.startswith("2026-05-15T12:00:00")


@patch("app.services.date_infer.datetime")
def test_through_n_days(mock_dt):
    fixed = datetime(2026, 5, 11, 9, 0, 0, tzinfo=MSK)
    _patch_now(fixed)(mock_dt)

    out = infer_due_iso_from_russian("через 3 дня сходить к врачу")
    assert out is not None
    assert out.startswith("2026-05-14T12:00:00")


@patch("app.services.date_infer.datetime")
def test_through_a_week(mock_dt):
    fixed = datetime(2026, 5, 11, 9, 0, 0, tzinfo=MSK)
    _patch_now(fixed)(mock_dt)

    out = infer_due_iso_from_russian("через 1 неделю сделать отчёт")
    assert out is not None
    assert out.startswith("2026-05-18T12:00:00")


@patch("app.services.date_infer.datetime")
def test_day_month_word(mock_dt):
    fixed = datetime(2026, 5, 11, 9, 0, 0, tzinfo=MSK)
    _patch_now(fixed)(mock_dt)

    out = infer_due_iso_from_russian("отметить 20 мая день рождения")
    assert out is not None
    assert out.startswith("2026-05-20T12:00:00")


@patch("app.services.date_infer.datetime")
def test_dotted_date(mock_dt):
    fixed = datetime(2026, 5, 11, 9, 0, 0, tzinfo=MSK)
    _patch_now(fixed)(mock_dt)

    out = infer_due_iso_from_russian("дедлайн 25.05")
    assert out is not None
    assert out.startswith("2026-05-25T12:00:00")


@patch("app.services.date_infer.datetime")
def test_no_marker_returns_none(mock_dt):
    fixed = datetime(2026, 5, 11, 9, 0, 0, tzinfo=MSK)
    _patch_now(fixed)(mock_dt)

    assert infer_due_iso_from_russian("купить молоко") is None
