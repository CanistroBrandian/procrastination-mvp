from __future__ import annotations

from datetime import datetime

from app.services.cron_match import cron_is_valid, cron_matches_window


def test_cron_matches_window_for_non_tick_minute() -> None:
    now = datetime(2026, 5, 14, 13, 10)
    assert cron_matches_window("7 * * * *", now, window_minutes=10) is True
    assert cron_matches_window("7 * * * *", now, window_minutes=2) is False


def test_cron_validation() -> None:
    assert cron_is_valid("*/10 * * * *") is True
    assert cron_is_valid("61 * * * *") is False
    assert cron_is_valid("*/x * * * *") is False
    assert cron_is_valid("* * *") is False
