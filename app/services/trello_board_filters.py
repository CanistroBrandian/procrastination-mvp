"""Общие правила: какие карточки считаем «на доске» для бота.

Активная задача: карточка не в архиве (closed=false) и не в закрытой колонке Trello,
и не в списке «Завершённые» (если он известен профилю).
"""

from __future__ import annotations

from typing import Any, Iterable


def archived_list_ids_from_trello_lists(lists: Iterable[dict[str, Any]]) -> set[str]:
    """id колонок, у которых в Trello closed=true (архивированная колонка)."""
    out: set[str] = set()
    for item in lists or []:
        if item.get("closed"):
            out.add(str(item.get("id") or ""))
    return {x for x in out if x}


def drop_cards_in_archived_lists(
    cards: Iterable[dict[str, Any]],
    lists: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Убирает карточки из архивированных колонок (список закрыт в Trello)."""
    archived = archived_list_ids_from_trello_lists(lists)
    if not archived:
        return list(cards or [])
    return [c for c in (cards or []) if str(c.get("idList") or "") not in archived]
