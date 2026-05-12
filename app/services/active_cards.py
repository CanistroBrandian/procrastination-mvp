"""Форматирование списка активных карточек на доске + фильтры по срокам."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from app.db.models import UserProfile
from app.services.date_infer import infer_due_iso_from_russian


# Даты «сегодня / завтра» считаем в одном часовом поясе (MVP).
_DISPLAY_TZ = ZoneInfo("Europe/Moscow")

_TELEGRAM_MAX = 3900  # запас под заголовок


@dataclass(frozen=True)
class CardsDisplayFilter:
    """Режим показа карточек."""

    kind: str  # all | due_within | today | tomorrow | week | overdue | on_date
    days: int | None = None  # для due_within и week (7)
    target_date: date | None = None  # для on_date


def parse_cards_nl_query(raw: str) -> CardsDisplayFilter | None:
    """Разбор текстовых запросов без /cards.

    Примеры:
      "Какие задачи у меня на сегодня?" -> today
      "Что по задачам завтра?" -> tomorrow
      "Покажи задачи" -> all
    """
    s = (raw or "").strip().lower()
    if not s:
        return None
    if s.startswith("/"):
        return None

    has_cards_topic = any(
        marker in s
        for marker in (
            "какие задачи",
            "что по задач",
            "покажи задач",
            "список задач",
            "какие дела",
            "что по делам",
            "что у меня",
            "что у них",
            "что нужно делать",
            "что осталось",
            "какой план",
            "план на",
        )
    )
    has_time_marker = any(t in s for t in ("сегодня", "завтра", "недел", "просроч"))
    has_task_word = any(w in s for w in ("задач", "дела", "дел", "план"))
    # Фразы вида «что нужно делать сегодня», «что осталось на сегодня».
    if not has_cards_topic and not (has_time_marker and has_task_word):
        return None

    if "сегодня" in s:
        return CardsDisplayFilter(kind="today")
    if "завтра" in s:
        return CardsDisplayFilter(kind="tomorrow")
    m_days = re.search(r"\bна\s+(\d{1,3})\s+дн", s)
    if m_days:
        n = int(m_days.group(1))
        if n > 0:
            return CardsDisplayFilter(kind="due_within", days=min(n, 365))
    inferred = infer_due_iso_from_russian(s)
    inferred_dt = _parse_due_iso(inferred) if inferred else None
    if inferred_dt is not None:
        return CardsDisplayFilter(kind="on_date", target_date=_due_date_local(inferred_dt))
    if "недел" in s:
        return CardsDisplayFilter(kind="week", days=7)
    if "просроч" in s:
        return CardsDisplayFilter(kind="overdue")
    return CardsDisplayFilter(kind="all")


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
        if flt.kind in {"today", "tomorrow", "due_within", "week", "overdue", "on_date"}:
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
    if flt.kind == "on_date":
        return flt.target_date is not None and d == flt.target_date
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
    if flt.kind == "on_date" and flt.target_date is not None:
        return f"На {flt.target_date.strftime('%d.%m.%Y')}"
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


def build_cards_overview_messages(
    profile: UserProfile,
    cards: Iterable[dict[str, Any]],
    list_id_to_name: dict[str, str],
    *,
    reference_date: date | None = None,
) -> list[str]:
    """Краткий обзор: сегодня / завтра / другой день, с учётом колонок."""
    today = reference_date or datetime.now(_DISPLAY_TZ).date()
    done_id = profile.trello_done_list_id

    active: list[dict[str, Any]] = []
    for c in cards:
        if c.get("closed"):
            continue
        if done_id and c.get("idList") == done_id:
            continue
        active.append(c)

    today_cards: list[dict[str, Any]] = []
    tomorrow_cards: list[dict[str, Any]] = []
    other_cards: list[dict[str, Any]] = []
    for c in active:
        dt = _parse_due_iso(c.get("due"))
        if dt is None:
            other_cards.append(c)
            continue
        d = _due_date_local(dt)
        if d == today:
            today_cards.append(c)
        elif d == today + timedelta(days=1):
            tomorrow_cards.append(c)
        else:
            other_cards.append(c)

    lines: list[str] = [
        "Обзор задач по датам:",
        f"Часовой пояс для «сегодня»: {_DISPLAY_TZ.key}",
    ]
    lines.extend(_render_bucket("На сегодня", today_cards, list_id_to_name))
    lines.extend(_render_bucket("На завтра", tomorrow_cards, list_id_to_name))
    lines.extend(_render_bucket("На другой день", other_cards, list_id_to_name))
    return _chunk_lines(lines)


def build_cards_by_days_window_messages(
    profile: UserProfile,
    cards: Iterable[dict[str, Any]],
    list_id_to_name: dict[str, str],
    days: int,
    *,
    reference_date: date | None = None,
) -> list[str]:
    """Выводит задачи по дням окна [сегодня .. сегодня+days-1]."""
    today = reference_date or datetime.now(_DISPLAY_TZ).date()
    done_id = profile.trello_done_list_id
    n = max(1, min(int(days or 1), 365))
    out_lines: list[str] = []

    active: list[dict[str, Any]] = []
    for c in cards:
        if c.get("closed"):
            continue
        if done_id and c.get("idList") == done_id:
            continue
        active.append(c)

    for offset in range(n):
        d = today + timedelta(days=offset)
        day_cards = []
        for c in active:
            dt = _parse_due_iso(c.get("due"))
            if dt is None:
                continue
            if _due_date_local(dt) == d:
                day_cards.append(c)
        if offset == 0:
            title = "На сегодня"
        elif offset == 1:
            title = "На завтра"
        else:
            title = f"На {d.strftime('%d.%m.%Y')}"
        out_lines.extend(_render_bucket(title, day_cards, list_id_to_name))

    return _chunk_lines(out_lines)


def _render_bucket(
    title: str,
    cards: list[dict[str, Any]],
    list_id_to_name: dict[str, str],
) -> list[str]:
    out: list[str] = [f"\n{title} ({len(cards)}):"]
    if not cards:
        out.append("— нет")
        return out

    by_list: dict[str, list[dict[str, Any]]] = {}
    for c in cards:
        lid = str(c.get("idList") or "")
        col = list_id_to_name.get(lid) or f"Колонка {lid[:8]}…"
        by_list.setdefault(col, []).append(c)

    for col_name in sorted(by_list.keys()):
        out.append(f"━━ {col_name} ━━")
        for c in sorted(by_list[col_name], key=_sort_key_card):
            name = (c.get("name") or "(без названия)").strip()
            badge = format_due_badge(c)
            url = (c.get("shortUrl") or "").strip()
            if url:
                out.append(f"• {name} — {badge}\n  {url}")
            else:
                out.append(f"• {name} — {badge}")
    return out


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
