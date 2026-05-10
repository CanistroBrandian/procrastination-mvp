"""Эвристики дат из русского текста (когда модель не заполнила due)."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

# Дни недели по нормальной форме (понедельник = 0).
_WEEKDAYS: dict[str, int] = {
    "понедельник": 0,
    "вторник": 1,
    "среда": 2,
    "среду": 2,
    "четверг": 3,
    "пятница": 4,
    "пятницу": 4,
    "суббота": 5,
    "субботу": 5,
    "воскресенье": 6,
}

# Месяцы для парсинга «10 мая» / «15 января».
_MONTHS: dict[str, int] = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "май": 5,
    "мая": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}


def calendar_context_for_prompt() -> str:
    now = datetime.now(MSK)
    return (
        f"Контекст календаря (Europe/Moscow): сейчас {now.strftime('%Y-%m-%d %H:%M')}, "
        f"сегодняшняя дата {now.date().isoformat()}. "
        "Если пользователь задаёт срок («сегодня», «завтра», «через неделю», «в пятницу», «10 мая», «15.05» и т.п.) — "
        "для create_card и update_card обязательно заполни поле due в ISO 8601 с часовым поясом "
        "(например 2026-05-10T12:00:00+03:00). Если из запроса нельзя однозначно понять срок, "
        "верни action_type=ask_for_clarification с question, не выдумывай дату."
    )


def _next_weekday(today: date, target_weekday: int, *, allow_today: bool = False) -> date:
    delta = (target_weekday - today.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return today + timedelta(days=delta)


def _build_iso(day: date, hour: int, minute: int) -> str:
    return datetime(day.year, day.month, day.day, hour, minute, 0, tzinfo=MSK).isoformat()


def infer_due_iso_from_russian(text: str, *, default_hour: int = 12, default_minute: int = 0) -> str | None:
    """
    Возвращает due в ISO 8601 с офсетом Москвы или None.

    Поддерживается:
        - сегодня / завтра / послезавтра
        - через N дней / через неделю / через месяц
        - в пятницу / в среду / в понедельник …
        - 10 мая / 15 января (текущий или следующий год)
        - 15.05 / 15.05.2026 / 2026-05-15
    """
    if not text or not text.strip():
        return None
    s = text.lower()
    now = datetime.now(MSK)
    today = now.date()

    if "послезавтра" in s:
        return _build_iso(today + timedelta(days=2), default_hour, default_minute)
    if "завтра" in s:
        return _build_iso(today + timedelta(days=1), default_hour, default_minute)
    if "сегодня" in s:
        return _build_iso(today, default_hour, default_minute)

    m = re.search(r"через\s+(\d+)\s+(день|дня|дней|недел[июя]|месяц[аев]?)", s)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("недел"):
            return _build_iso(today + timedelta(weeks=num), default_hour, default_minute)
        if unit.startswith("месяц"):
            return _build_iso(today + timedelta(days=30 * num), default_hour, default_minute)
        return _build_iso(today + timedelta(days=num), default_hour, default_minute)
    if re.search(r"через\s+неделю", s):
        return _build_iso(today + timedelta(weeks=1), default_hour, default_minute)
    if re.search(r"через\s+месяц", s):
        return _build_iso(today + timedelta(days=30), default_hour, default_minute)

    for word, idx in _WEEKDAYS.items():
        if re.search(rf"\bв(?:о)?\s+{word}\b", s) or re.search(rf"\b{word}\b", s):
            return _build_iso(_next_weekday(today, idx), default_hour, default_minute)

    m = re.search(r"\b(\d{1,2})\s+([а-яё]+)\b", s)
    if m:
        d = int(m.group(1))
        month_word = m.group(2)
        month = next(
            (num for prefix, num in _MONTHS.items() if month_word.startswith(prefix)),
            None,
        )
        if month and 1 <= d <= 31:
            year = today.year
            if (month, d) < (today.month, today.day):
                year += 1
            try:
                return _build_iso(date(year, month, d), default_hour, default_minute)
            except ValueError:
                pass

    m = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?\b", s)
    if m:
        d = int(m.group(1))
        mo = int(m.group(2))
        y = m.group(3)
        year = int(y) if y else today.year
        if year < 100:
            year += 2000
        try:
            day = date(year, mo, d)
            if not y and day < today:
                day = date(year + 1, mo, d)
            return _build_iso(day, default_hour, default_minute)
        except ValueError:
            pass

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)
    if m:
        try:
            return _build_iso(date(int(m.group(1)), int(m.group(2)), int(m.group(3))), default_hour, default_minute)
        except ValueError:
            pass

    return None
