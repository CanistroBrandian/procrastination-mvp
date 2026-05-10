"""Форматирование списка активных карточек на доске + фильтры по срокам."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from app.db.models import UserProfile


# Даты «сегодня / завтра» считаем в одном часовом поясе (MVP).
_DISPLAY_TZ = ZoneInfo("Europe/Moscow")

_TELEGRAM_MAX = 3900  # запас под заголовок


@dataclass(frozen=True)
class CardsDisplayFilter:
    """Режим показа карточек."""

    kind: str  # all | due_within | today | tomorrow | week | overdue
    days: int | None = None  # для due_within и week (7)


def parse_cards_command_arguments(raw: str) -> CardsDisplayFilter | None:
    """Разбор аргументов после `/cards ...`.

    Примеры:
      /cards  → all
      /cards 7  → срок в ближайшие 7 календарных дней (включая сегодня)
      /cards неделя | week  → то же, 7 дней
      /cards сегодня | today
      /cards завтра | tomorrow
      /cards просрочка | overdue | просроченные
    """
    s = (raw or "").strip()
    if not s.startswith("/cards"):
        return None
    rest = s[len("/cards") :].strip()
    if not rest:
        return CardsDisplayFilter(kind="all")

    token = rest.split()[0].lower().strip()

    if token in {"сегодня", "today"}:
        return CardsDisplayFilter(kind="today")
    if token in {"завтра", "tomorrow"}:
        return CardsDisplayFilter(kind="tomorrow")
    if token in {"неделя", "week", "на неделю"}:
        return CardsDisplayFilter(kind="week", days=7)
    if token in {"просрочка", "просроченные", "просрочено", "overdue"}:
        return CardsDisplayFilter(kind="overdue")

    m = re.fullmatch(r"(\d{1,3})", token)
    if m:
        n = int(m.group(1))
        if n == 0:
            return None
        return CardsDisplayFilter(kind="due_within", days=min(n, 365))

    return None


def _parse_due_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    # Trello отдаёт ISO с Z или оффсетом
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_DISPLAY_TZ)
        return dt.astimezone(_DISPLAY_TZ)
    except ValueError:
        return None


def _due_date_local(dt: datetime) -> date:
    return dt.astimezone(_DISPLAY_TZ).date()


def _card_matches_filter(
    card: dict[str, Any],
    flt: CardsDisplayFilter,
    *,
    today: date,
) -> bool:
    """Подходит ли карточка под фильтр по дедлайну."""
    due_raw = card.get("due")
    due_complete = bool(card.get("dueComplete"))
    dt = _parse_due_iso(due_raw) if due_raw else None

    if flt.kind == "all":
        return True

    if due_complete and flt.kind != "all":
        # Выполненный дедлайн не показываем в окнах по датам (кроме all — там всё равно покажем метку)
        if flt.kind in {"today", "tomorrow", "due_within", "week", "overdue"}:
            return False

    if flt.kind == "overdue":
        if not dt or due_complete:
            return False
        return _due_date_local(dt) < today

    if not dt:
        return False
    d = _due_date_local(dt)

    if flt.kind == "today":
        return d == today
    if flt.kind == "tomorrow":
        return d == today + timedelta(days=1)
    if flt.kind in ("due_within", "week"):
        n = flt.days or 7
        end = today + timedelta(days=n - 1)
        return today <= d <= end

    return True


def _sort_key_card(card: dict[str, Any]) -> tuple[str, str]:
    dt = _parse_due_iso(card.get("due"))
    name = (card.get("name") or "").lower()
    if dt is None:
        return ("9999-12-31", name)
    return (_due_date_local(dt).isoformat(), name)


def _filter_title(flt: CardsDisplayFilter) -> str:
    if flt.kind == "all":
        return "Все активные карточки"
    if flt.kind == "today":
        return "На сегодня"
    if flt.kind == "tomorrow":
        return "На завтра"
    if flt.kind == "week":
        return "На ближайшие 7 дней"
    if flt.kind == "due_within":
        return f"На ближайшие {flt.days} дн."
    if flt.kind == "overdue":
        return "Просроченные (по дедлайну)"
    return "Карточки"


def format_due_badge(card: dict[str, Any]) -> str:
    """Короткая метка срока для строки списка."""
    due_complete = bool(card.get("dueComplete"))
    dt = _parse_due_iso(card.get("due"))
    if due_complete:
        return "✓ срок выполнен"
    if not dt:
        return "без срока"
    d = _due_date_local(dt)
    return d.strftime("%d.%m.%Y")


def build_active_cards_messages(
    profile: UserProfile,
    cards: Iterable[dict[str, Any]],
    list_id_to_name: dict[str, str],
    flt: CardsDisplayFilter,
    *,
    reference_date: date | None = None,
) -> list[str]:
    """Собирает одно или несколько сообщений для Telegram.

    Активная карточка: не в архиве карточки (closed=false), не в архивной колонке
    (отфильтруйте до вызова через `drop_cards_in_archived_lists`), и не в колонке
    «Завершённые», если для профиля задан trello_done_list_id.
    """
    today = reference_date or datetime.now(_DISPLAY_TZ).date()
    done_id = profile.trello_done_list_id

    raw_list = list(cards)
    active: list[dict[str, Any]] = []
    for c in raw_list:
        if c.get("closed"):
            continue
        lid = c.get("idList")
        if done_id and lid == done_id:
            continue
        if not _card_matches_filter(c, flt, today=today):
            continue
        active.append(c)

    # Сортировка: по дате due (нет срока — в конец), затем по имени
    def sort_key(card: dict[str, Any]) -> tuple[int, str]:
        dt = _parse_due_iso(card.get("due"))
        if dt is None or card.get("dueComplete"):
            return (1, (card.get("name") or "").lower())
        return (0, _due_date_local(dt).isoformat() + (card.get("name") or "").lower())

    active.sort(key=_sort_key_card)

    title = _filter_title(flt)
    header = f"{title} ({len(active)}):\nЧасовой пояс для «сегодня»: {_DISPLAY_TZ.key}\n"

    if not active:
        return [header + "\nНичего не найдено."]

    # Группировка по колонке
    by_list: dict[str, list[dict[str, Any]]] = {}
    for c in active:
        lid = str(c.get("idList") or "")
        col = list_id_to_name.get(lid) or f"Колонка {lid[:8]}…"
        by_list.setdefault(col, []).append(c)

    lines: list[str] = [header]
    for col_name in sorted(by_list.keys()):
        lines.append(f"\n━━ {col_name} ━━")
        for c in by_list[col_name]:
            name = (c.get("name") or "(без названия)").strip()
            badge = format_due_badge(c)
            url = (c.get("shortUrl") or "").strip()
            if url:
                lines.append(f"• {name} — {badge}\n  {url}")
            else:
                lines.append(f"• {name} — {badge}")

    # Разбивка на сообщения по лимиту Telegram
    return _chunk_lines(lines)


def _chunk_lines(lines: list[str]) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        line_len = len(line) + 1
        if buf and size + line_len > _TELEGRAM_MAX:
            chunks.append("\n".join(buf))
            buf = [line]
            size = len(line)
        else:
            buf.append(line)
            size += line_len
    if buf:
        chunks.append("\n".join(buf))
    return chunks
