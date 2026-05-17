from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class CronFields:
    minute: str
    hour: str
    day: str
    month: str
    weekday: str


def _match_field(value: int, expr: str, *, min_v: int, max_v: int) -> bool:
    token = (expr or "").strip()
    if token == "*":
        return True
    parts = [p.strip() for p in token.split(",") if p.strip()]
    for part in parts:
        if part == "*":
            return True
        if part.startswith("*/"):
            try:
                step = int(part[2:])
            except ValueError:
                continue
            if step <= 0:
                continue
            if value % step == 0:
                return True
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(left)
                end = int(right)
            except ValueError:
                continue
            if start <= value <= end:
                return True
            continue
        try:
            n = int(part)
        except ValueError:
            continue
        if min_v <= n <= max_v and n == value:
            return True
    return False


def _is_field_valid(expr: str, *, min_v: int, max_v: int) -> bool:
    token = (expr or "").strip()
    if token == "*":
        return True
    parts = [p.strip() for p in token.split(",") if p.strip()]
    if not parts:
        return False
    for part in parts:
        if part == "*":
            continue
        if part.startswith("*/"):
            try:
                step = int(part[2:])
            except ValueError:
                return False
            if step <= 0:
                return False
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(left)
                end = int(right)
            except ValueError:
                return False
            if start > end or start < min_v or end > max_v:
                return False
            continue
        try:
            n = int(part)
        except ValueError:
            return False
        if n < min_v or n > max_v:
            return False
    return True


def cron_matches_now(expr: str, now: datetime) -> bool:
    chunks = [x.strip() for x in (expr or "").split()]
    if len(chunks) != 5:
        return False
    fields = CronFields(*chunks)
    # Python: Monday=0..Sunday=6; classic cron: Sunday=0 or 7.
    cron_weekday = (now.weekday() + 1) % 7
    return (
        _match_field(now.minute, fields.minute, min_v=0, max_v=59)
        and _match_field(now.hour, fields.hour, min_v=0, max_v=23)
        and _match_field(now.day, fields.day, min_v=1, max_v=31)
        and _match_field(now.month, fields.month, min_v=1, max_v=12)
        and _match_field(cron_weekday, fields.weekday, min_v=0, max_v=7)
    )


def cron_is_valid(expr: str) -> bool:
    chunks = [x.strip() for x in (expr or "").split()]
    if len(chunks) != 5:
        return False
    minute, hour, day, month, weekday = chunks
    return (
        _is_field_valid(minute, min_v=0, max_v=59)
        and _is_field_valid(hour, min_v=0, max_v=23)
        and _is_field_valid(day, min_v=1, max_v=31)
        and _is_field_valid(month, min_v=1, max_v=12)
        and _is_field_valid(weekday, min_v=0, max_v=7)
    )


def normalize_cron_or_default(expr: str | None, default_expr: str) -> tuple[str, bool]:
    """
    Returns a safe cron expression and whether fallback was used.
    """
    raw = (expr or "").strip()
    if cron_is_valid(raw):
        return raw, False
    fallback = (default_expr or "").strip()
    if cron_is_valid(fallback):
        return fallback, True
    return "* * * * *", True


def cron_matches_window(expr: str, now: datetime, *, window_minutes: int) -> bool:
    window = max(0, int(window_minutes))
    for shift in range(window + 1):
        probe = now - timedelta(minutes=shift)
        if cron_matches_now(expr, probe):
            return True
    return False
