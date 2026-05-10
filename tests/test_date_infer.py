from datetime import datetime
from unittest.mock import patch

from zoneinfo import ZoneInfo

from app.services.date_infer import infer_due_iso_from_russian

MSK = ZoneInfo("Europe/Moscow")


def test_infer_no_markers_returns_none():
    assert infer_due_iso_from_russian("купить молоко") is None


@patch("app.services.date_infer.datetime")
def test_infer_zavtra_uses_next_calendar_day(mock_datetime):
    """Фиксируем «сейчас», проверяем ISO на следующий день (МСК)."""
    real_datetime = __import__("datetime").datetime
    mock_datetime.timedelta = __import__("datetime").timedelta

    fixed_now = real_datetime(2026, 5, 9, 22, 30, 0, tzinfo=MSK)

    def fake_now(tz=None):
        assert tz is MSK
        return fixed_now

    mock_datetime.now.side_effect = fake_now
    mock_datetime.side_effect = lambda *a, **k: real_datetime(*a, **k)

    out = infer_due_iso_from_russian("задача на завтра помыть машину")
    assert out is not None
    assert out.startswith("2026-05-10T12:00:00")


@patch("app.services.date_infer.datetime")
def test_poslezavtra_before_zavtra_substring(mock_datetime):
    real_datetime = __import__("datetime").datetime
    mock_datetime.timedelta = __import__("datetime").timedelta
    fixed_now = real_datetime(2026, 5, 9, 10, 0, 0, tzinfo=MSK)
    mock_datetime.now.side_effect = lambda tz=None: fixed_now
    mock_datetime.side_effect = lambda *a, **k: real_datetime(*a, **k)

    out = infer_due_iso_from_russian("сделать послезавтра важное")
    assert out is not None
    assert out.startswith("2026-05-11T12:00:00")
