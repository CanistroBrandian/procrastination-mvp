"""Фаззи-поиск пункта чеклиста по русскому тексту пользователя.

Без новых зависимостей — комбинируем difflib и грубый «стеммер»:
обрезаем 2 буквы у длинных русских слов, чтобы «помыл/помыть/помыли» совпадали.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

# Стоп-слова и предлоги — мешают подсчёту пересечения токенов.
_STOPWORDS = {
    "и", "в", "на", "по", "из", "за", "с", "со", "о", "об", "у", "к", "до", "от",
    "что", "чтобы", "это", "так", "ещё", "ну", "же", "мне", "тебя", "его", "её",
    "я", "ты", "он", "она", "мы", "вы", "они",
    "был", "была", "было", "были",
    "уже", "там", "тут", "вот",
}

_PUNCT_RX = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACES_RX = re.compile(r"\s+")


@dataclass(frozen=True)
class CheckItemRef:
    card_id: str
    card_name: str
    list_id: str | None
    checklist_id: str
    checklist_name: str
    check_item_id: str
    item_name: str
    state: str  # "complete" | "incomplete"
    short_url: str | None = None


@dataclass(frozen=True)
class CardRef:
    card_id: str
    card_name: str
    list_id: str | None
    short_url: str | None
    incomplete_items: tuple[str, ...]  # имена незавершённых пунктов всех чеклистов
    total_items: int


def normalize(text: str) -> str:
    s = (text or "").lower().replace("ё", "е")
    s = _PUNCT_RX.sub(" ", s)
    return _SPACES_RX.sub(" ", s).strip()


def tokens(text: str) -> list[str]:
    return [
        t
        for t in normalize(text).split()
        if t and len(t) > 1 and t not in _STOPWORDS
    ]


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _word_similar(a: str, b: str, *, prefix_min: int = 4) -> bool:
    """Слова русского достаточно близки: одинаковые, общий префикс ≥ N или high SequenceMatcher."""
    if a == b:
        return True
    if len(a) >= prefix_min and len(b) >= prefix_min and _common_prefix_len(a, b) >= prefix_min:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.78


def _overlap_score(qt: list[str], ct: list[str]) -> float:
    if not qt or not ct:
        return 0.0
    matched = 0
    for w in qt:
        if any(_word_similar(w, x) for x in ct):
            matched += 1
    return matched / max(len(qt), len(ct))


def score(query: str, candidate: str) -> float:
    qn = normalize(query)
    cn = normalize(candidate)
    if not qn or not cn:
        return 0.0
    seq = SequenceMatcher(None, qn, cn).ratio()
    qt = tokens(query)
    ct = tokens(candidate)
    if not qt or not ct:
        return seq
    overlap = _overlap_score(qt, ct)
    return 0.4 * seq + 0.6 * overlap


def flatten_incomplete_items(cards: Iterable[dict[str, Any]]) -> list[CheckItemRef]:
    out: list[CheckItemRef] = []
    for card in cards or []:
        if card.get("closed"):
            continue
        for cl in card.get("checklists") or []:
            for ci in cl.get("checkItems") or []:
                if (ci.get("state") or "incomplete") != "incomplete":
                    continue
                out.append(
                    CheckItemRef(
                        card_id=card.get("id", ""),
                        card_name=card.get("name", ""),
                        list_id=card.get("idList"),
                        checklist_id=cl.get("id", ""),
                        checklist_name=cl.get("name", ""),
                        check_item_id=ci.get("id", ""),
                        item_name=ci.get("name", ""),
                        state=ci.get("state") or "incomplete",
                        short_url=card.get("shortUrl"),
                    )
                )
    return out


def best_matches(
    query: str,
    items: Iterable[CheckItemRef],
    *,
    top_k: int = 5,
    min_score: float = 0.35,
) -> list[tuple[CheckItemRef, float]]:
    scored = [(it, score(query, it.item_name)) for it in items]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(it, s) for it, s in scored[:top_k] if s >= min_score]


def confident_unique(matches: list[tuple[CheckItemRef, float]], *, gap: float = 0.18) -> CheckItemRef | None:
    """Уникальный кандидат, если он один или существенно опережает второй."""
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]
    return matches[0][0] if (matches[0][1] - matches[1][1]) >= gap else None


def cards_from_payload(cards: Iterable[dict[str, Any]]) -> list[CardRef]:
    out: list[CardRef] = []
    for card in cards or []:
        if card.get("closed"):
            continue
        items_total = 0
        incomplete: list[str] = []
        for cl in card.get("checklists") or []:
            for ci in cl.get("checkItems") or []:
                items_total += 1
                if (ci.get("state") or "incomplete") != "complete":
                    incomplete.append(str(ci.get("name") or ""))
        out.append(
            CardRef(
                card_id=card.get("id", ""),
                card_name=card.get("name", ""),
                list_id=card.get("idList"),
                short_url=card.get("shortUrl"),
                incomplete_items=tuple(incomplete),
                total_items=items_total,
            )
        )
    return out


def best_card_matches(
    query: str,
    cards: Iterable[CardRef],
    *,
    top_k: int = 5,
    min_score: float = 0.35,
) -> list[tuple[CardRef, float]]:
    scored = [(c, score(query, c.card_name)) for c in cards]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(c, s) for c, s in scored[:top_k] if s >= min_score]


def confident_unique_card(matches: list[tuple[CardRef, float]], *, gap: float = 0.18) -> CardRef | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]
    return matches[0][0] if (matches[0][1] - matches[1][1]) >= gap else None
