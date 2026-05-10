"""Семантический поиск карточек на доске Trello.

Сценарий: пользователь ссылается на карточку в свободной форме
(«машину обслужили, можно закрывать», «отчёт уже готов, перенеси в done»),
а сервер должен понять, какую именно карточку он имеет в виду.

Подход — гибридный, на двух уровнях:
1) `app.services.checklist_match.best_card_matches` — быстрый fuzzy
   по названию (морфология русского + SequenceMatcher).
2) `AgentService.search_cards_semantic` — LLM-агент семантического поиска.
   Получает реплику пользователя + сводки карточек (название, чеклисты, статус)
   и возвращает релевантные id с обоснованием.

`merge_search_results` объединяет оба источника в один ранжированный список
`CardSearchHit`, к которому оркестратор обращается дальше.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.services.checklist_match import CardRef


@dataclass(frozen=True)
class CardSearchHit:
    """Кандидат-карточка от поисковика. score уже агрегирован."""

    card_id: str
    card_name: str
    score: float
    reason: str
    incomplete_items: tuple[str, ...]
    total_items: int
    list_id: str | None


def format_cards_for_search(
    cards: Iterable[CardRef], *, limit: int = 25,
) -> list[dict[str, object]]:
    """Компактные сводки карточек для LLM-промпта (экономим токены)."""
    out: list[dict[str, object]] = []
    for c in list(cards)[:limit]:
        out.append(
            {
                "id": c.card_id,
                "name": c.card_name,
                "incomplete_items": list(c.incomplete_items[:5]),
                "total_items": c.total_items,
            }
        )
    return out


def merge_search_results(
    fuzzy: list[tuple[CardRef, float]],
    semantic: list[CardSearchHit],
    cards: list[CardRef],
    *,
    fuzzy_weight: float = 0.4,
    semantic_weight: float = 0.6,
    keep_top: int = 5,
    min_score: float = 0.30,
) -> list[CardSearchHit]:
    """Объединяет результаты fuzzy и семантического поиска по card_id.

    fuzzy без semantic — кандидаты с маленьким весом (название зацепилось,
    но смысл не подтверждён). semantic без fuzzy — модель нашла связь по
    смыслу (например, «машину обслужили» → «обслуживание машины» при сильно
    разных формах слов). Оба сразу — самый сильный кандидат.
    """
    by_id_card = {c.card_id: c for c in cards}
    scores: dict[str, dict[str, object]] = {}

    for ref, fscore in fuzzy:
        scores[ref.card_id] = {
            "fuzzy": float(fscore),
            "semantic": 0.0,
            "reason": "по названию",
        }
    for hit in semantic:
        bucket = scores.setdefault(
            hit.card_id, {"fuzzy": 0.0, "semantic": 0.0, "reason": ""},
        )
        bucket["semantic"] = float(hit.score)
        if hit.reason:
            bucket["reason"] = hit.reason

    # Если один из источников пуст — отдаём весь вес другому, чтобы кандидаты
    # не отсекались min_score просто из-за половинного веса.
    if not semantic:
        fuzzy_weight = 1.0
        semantic_weight = 0.0
    elif not fuzzy:
        fuzzy_weight = 0.0
        semantic_weight = 1.0

    out: list[CardSearchHit] = []
    for cid, bucket in scores.items():
        ref = by_id_card.get(cid)
        if ref is None:
            continue
        score = (
            float(bucket["fuzzy"]) * fuzzy_weight
            + float(bucket["semantic"]) * semantic_weight
        )
        if score < min_score:
            continue
        reason = str(bucket.get("reason") or "по названию")
        out.append(
            CardSearchHit(
                card_id=ref.card_id,
                card_name=ref.card_name,
                score=score,
                reason=reason,
                incomplete_items=ref.incomplete_items,
                total_items=ref.total_items,
                list_id=ref.list_id,
            )
        )

    out.sort(key=lambda h: h.score, reverse=True)
    return out[:keep_top]


def confident_unique_hit(
    hits: list[CardSearchHit], *, gap: float = 0.18,
) -> CardSearchHit | None:
    """Один уверенный кандидат, либо явно лидирующий по score."""
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    return hits[0] if (hits[0].score - hits[1].score) >= gap else None
