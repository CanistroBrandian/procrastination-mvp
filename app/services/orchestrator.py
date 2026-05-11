from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.core.errors import BoardNotLinkedError, OrchestratorValidationError


logger = logging.getLogger(__name__)
from app.db.models import Persona, UserProfile
from app.db.repositories import ClarificationRepository
from app.integrations.trello import TrelloClient
from app.schemas.actions import AgentAction, AgentResult
from app.services.agent import AgentService
from app.services.card_search import (
    CardSearchHit,
    confident_unique_hit,
    format_cards_for_search,
    merge_search_results,
)
from app.services.checklist_match import (
    CardRef,
    CheckItemRef,
    best_card_matches,
    best_matches,
    cards_from_payload,
    confident_unique,
    confident_unique_card,
    flatten_incomplete_items,
)
from app.services.date_infer import infer_due_iso_from_russian
from app.services.intent_rules import quick_classify_intent
from app.services.trello_board_filters import drop_cards_in_archived_lists


@dataclass
class OrchestratorResult:
    text: str


def _split_checklist_items(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in re.split(r"[\n;|]+", raw) if p.strip()]


def _normalize_guillemets_for_inference(text: str) -> str:
    """Whisper/Телеграм иногда отдают „..." или Unicode-кавычки вместо «…»."""
    if not text:
        return text
    s = text
    for src, dst in (
        ("\u201c", "«"),
        ("\u201d", "»"),
        ("\u201e", "«"),
        ("\u00ab", "«"),
        ("\u00bb", "»"),
        ("\u2039", "«"),
        ("\u203a", "»"),
    ):
        s = s.replace(src, dst)
    return s


def _infer_card_title_from_user_text(text: str) -> str | None:
    """Вытаскивает название карточки из разговорной фразы («…», «называется …»)."""
    s = _normalize_guillemets_for_inference((text or "").strip())
    if not s:
        return None
    patterns = (
        r"называется\s*«([^»]+)»",
        r"называется\s*\"([^\"]+)\"",
        r"к\s+карточк[аеу]\s*«([^»]+)»",
        r"в\s+задач[аеу]\s*«([^»]+)»",
        r"задач[аеиу]\s*,?\s*которая\s+называется\s*«([^»]+)»",
    )
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            t = m.group(1).strip()
            if len(t) >= 2:
                return t
    m = re.search(r"«([^»]{2,120})»", s)
    if m:
        return m.group(1).strip()
    return None


_RU_INDEX_WORDS: dict[str, int] = {
    "один": 1, "одну": 1, "одна": 1, "одно": 1, "одного": 1,
    "первый": 1, "первая": 1, "первое": 1, "первую": 1, "первого": 1, "первой": 1,
    "два": 2, "две": 2, "двух": 2, "двое": 2,
    "второй": 2, "вторая": 2, "второе": 2, "вторую": 2, "второго": 2,
    "три": 3, "трёх": 3, "трех": 3, "трое": 3,
    "третий": 3, "третья": 3, "третье": 3, "третью": 3, "третьего": 3, "третьей": 3,
    "четыре": 4, "четырёх": 4, "четырех": 4,
    "четвертый": 4, "четвёртый": 4, "четвертая": 4, "четвёртая": 4,
    "четвертое": 4, "четвёртое": 4, "четвертую": 4, "четвёртую": 4,
    "пять": 5, "пятый": 5, "пятая": 5, "пятое": 5, "пятую": 5,
    "шесть": 6, "шестой": 6, "шестая": 6, "шестое": 6, "шестую": 6,
    "семь": 7, "седьмой": 7, "седьмая": 7, "седьмое": 7, "седьмую": 7,
    "восемь": 8, "восьмой": 8, "восьмая": 8, "восьмое": 8, "восьмую": 8,
    "девять": 9, "девятый": 9, "девятая": 9, "девятое": 9, "девятую": 9,
    "десять": 10, "десятый": 10, "десятая": 10, "десятое": 10, "десятую": 10,
}


def _extract_index_from_reply(reply: str, max_idx: int) -> int | None:
    """Понимает «1», «номер 2», «под номером один», «выбираю первую» и т.п."""
    if max_idx <= 0:
        return None
    s = (reply or "").strip().lower().replace("ё", "е")
    if not s:
        return None
    m = re.search(r"\b(\d{1,2})\b", s)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= max_idx:
            return idx
    for token in re.findall(r"[а-яa-z]+", s):
        normalized = token.replace("ё", "е")
        idx = _RU_INDEX_WORDS.get(normalized)
        if idx is not None and 1 <= idx <= max_idx:
            return idx
    return None


def _checklist_add_card_search_queries(user_text: str, action: AgentAction) -> list[str]:
    """Запросы для поиска карточки при добавлении в чеклист.

    Явное «название в кавычках» в реплике надёжнее поля card_name от LLM: модель часто
    кладёт туда предметы чеклиста («английский», «география»), из‑за чего fuzzy не находит карточку.
    """
    ut = (user_text or "").strip()
    inferred = _infer_card_title_from_user_text(ut)
    cn = (action.card_name or "").strip()
    out: list[str] = []
    seen_lower: set[str] = set()

    def push(q: str) -> None:
        q = q.strip()
        if len(q) < 2:
            return
        low = q.lower()
        if low in seen_lower:
            return
        seen_lower.add(low)
        out.append(q)

    if inferred:
        push(inferred)
    if cn:
        push(cn)
    if ut and len(ut) <= 800:
        push(ut)
    return out


class TaskOrchestrator:
    def __init__(
        self,
        agent_service: AgentService,
        trello_client: TrelloClient,
        clarification_repo: ClarificationRepository,
    ):
        self.agent_service = agent_service
        self.trello_client = trello_client
        self.clarification_repo = clarification_repo

    @staticmethod
    def _apply_due_fallback(user_text: str, action: AgentAction) -> AgentAction:
        """Если модель не заполнила due, но в тексте есть дата — выставляем срок."""
        if action.due is not None:
            return action
        if action.action_type not in ("create_card", "update_card"):
            return action
        inferred = infer_due_iso_from_russian(user_text)
        if not inferred:
            return action
        return action.model_copy(update={"due": inferred})

    @staticmethod
    def _ensure_create_card_complete(user_text: str, action: AgentAction) -> AgentAction:
        """Use-case 1: для create_card обязательны card_name и due — иначе уточняем."""
        if action.action_type != "create_card":
            return action
        missing: list[str] = []
        if not action.card_name or not action.card_name.strip():
            missing.append("card_name")
        if not action.due:
            missing.append("due")
        if not missing:
            return action

        question_parts: list[str] = []
        if "card_name" in missing:
            question_parts.append("как назвать задачу")
        if "due" in missing:
            question_parts.append("к какому сроку её сделать (например «завтра в 18:00» или «10 мая»)")
        question = "Уточните, пожалуйста, " + " и ".join(question_parts) + "."

        base_meta = dict(action.metadata or {})
        base_meta.update(
            {
                "after_clarification": "create_card",
                "missing_fields": missing,
                "original_user_text": user_text,
                "draft_card_name": action.card_name,
                "draft_card_description": action.card_description,
                "draft_list_name": action.list_name,
                "draft_due": action.due,
                "draft_start": action.start,
                "draft_position": action.position,
                "draft_checklist_name": action.checklist_name,
                "draft_checklist_item": action.checklist_item,
            }
        )
        return action.model_copy(
            update={
                "action_type": "ask_for_clarification",
                "question": question,
                "metadata": base_meta,
            }
        )

    @staticmethod
    def _ensure_checklist_has_card_or_ask(original_user_text: str, action: AgentAction) -> AgentAction:
        """create_checklist_item без card_id → уточнение, контекст в metadata."""
        if action.action_type != "create_checklist_item":
            return action
        if action.card_id and action.checklist_item:
            return action
        base_meta = dict(action.metadata or {})
        payload = {
            "after_clarification": "create_checklist_item",
            "original_user_text": original_user_text,
            "pending_checklist_name": action.checklist_name,
            "pending_checklist_item": action.checklist_item,
        }
        if not action.checklist_item:
            return action.model_copy(
                update={
                    "action_type": "ask_for_clarification",
                    "question": (
                        "Что именно добавить в чеклист? Перечислите пункты. "
                        "Если чеклист к уже существующей карточке — пришлите также ссылку или id карточки."
                    ),
                    "metadata": {**base_meta, **payload},
                }
            )
        extra = ""
        if action.card_name:
            extra = (
                f"Не нашёл на доске карточку по названию «{action.card_name}». "
                "Пришлите ссылку на карточку "
            )
        else:
            extra = "Чтобы добавить пункты в чеклист, нужна карточка в Trello. Пришлите ссылку на карточку "
        return action.model_copy(
            update={
                "action_type": "ask_for_clarification",
                "question": (
                    f"{extra}"
                    "(https://trello.com/c/…) или id карточки (24 символа из URL)."
                ),
                "metadata": {**base_meta, **payload},
            }
        )

    @staticmethod
    def _checklist_resume_prompt(meta: dict[str, object], clarification_reply: str) -> str:
        orig = meta.get("original_user_text") or ""
        items = meta.get("pending_checklist_item") or ""
        cl_name = meta.get("pending_checklist_name") or ""
        return (
            "[Продолжение задачи с чеклистом]\n"
            f"Исходный запрос пользователя: {orig}\n"
            f"Название чеклиста (если было): {cl_name}\n"
            f"Пункты чеклиста: {items}\n"
            f"Пользователь указал карточку или уточнение: {clarification_reply}\n"
            "Верни JSON create_checklist_item: card_id если есть ссылка/id; иначе card_name — название карточки для поиска; "
            "сохрани checklist_item. Если ничего не ясно — ask_for_clarification."
        )

    @staticmethod
    def _create_card_resume_prompt(meta: dict[str, object], clarification_reply: str) -> str:
        orig = meta.get("original_user_text") or ""
        draft_name = meta.get("draft_card_name") or ""
        draft_due = meta.get("draft_due") or ""
        draft_items = meta.get("draft_checklist_item") or ""
        missing = ", ".join(meta.get("missing_fields") or []) or "?"
        return (
            "[Продолжение создания карточки]\n"
            f"Исходный запрос: {orig}\n"
            f"Уже извлечено — название: {draft_name!r}, срок (due): {draft_due!r}, "
            f"подзадачи: {draft_items!r}.\n"
            f"Не хватало: {missing}.\n"
            f"Ответ пользователя на уточнение: {clarification_reply}\n"
            "Верни JSON для create_card: подставь недостающее (card_name и/или due в ISO 8601), "
            "сохрани прежние значения, не выдумывай новые подзадачи. Если срок всё ещё не ясен — ask_for_clarification."
        )

    async def _resolve_add_checklist_card(
        self,
        profile: UserProfile,
        user_text: str,
        action: AgentAction,
    ) -> tuple[AgentAction, str | None]:
        """Если пытаемся добавить пункты в чеклист без card_id — ищем карточку по названию на доске."""
        if action.action_type != "create_checklist_item":
            return action, None
        if action.card_id or not (action.checklist_item or "").strip():
            return action, None
        if not profile.trello_board_id:
            return action, None

        queries = _checklist_add_card_search_queries(user_text, action)
        if not queries:
            return action, None

        lists_raw = await self.trello_client.list_lists(profile.trello_board_id)
        payload = await self.trello_client.list_board_cards_with_checklists(profile.trello_board_id)
        payload = drop_cards_in_archived_lists(payload, lists_raw)
        cards = cards_from_payload(payload)
        # Не отбрасываем колонку «Готово»: пользователь может ссылаться на карточку там,
        # и пункты чеклиста всё равно имеет смысл добавить.

        hits: list[CardSearchHit] = []
        q_used = queries[0]
        for q in queries:
            hits = await self._search_cards_hybrid(profile, q, cards)
            if hits:
                q_used = q
                break
        if not hits:
            logger.info("add_checklist: no card hits after queries=%r", queries)
            return action, None

        winner = confident_unique_hit(hits)
        if winner is not None:
            ref = next((c for c in cards if c.card_id == winner.card_id), None)
            if ref is not None:
                logger.info(
                    "add_checklist: unique match card_id=%s name=%r",
                    ref.card_id,
                    ref.card_name,
                )
                return action.model_copy(update={"card_id": ref.card_id}), None

        cands: list[dict[str, str]] = [
            {"card_id": h.card_id, "card_name": h.card_name} for h in hits[:5]
        ]
        listing = "\n".join(f"{i + 1}) {c['card_name']}" for i, c in enumerate(cands))
        question = (
            f"Нашёл несколько карточек по запросу «{q_used}». "
            f"В какую добавить пункты чеклиста? Укажите номер:\n{listing}"
        )
        return (
            action.model_copy(
                update={
                    "action_type": "ask_for_clarification",
                    "question": question,
                    "metadata": {
                        **(action.metadata or {}),
                        "after_clarification": "add_checklist_pick_card",
                        "pending_checklist_item": action.checklist_item,
                        "pending_checklist_name": action.checklist_name or "Шаги",
                        "candidates": cands,
                    },
                }
            ),
            None,
        )

    async def _resolve_complete_task(
        self,
        profile: UserProfile,
        user_text: str,
        action: AgentAction,
    ) -> tuple[AgentAction, str | None]:
        """Ищем подходящий пункт чеклиста.

        - Один уверенный кандидат → action заменяется на update_checklist_item;
          возвращаем сопровождающий текст.
        - Несколько похожих → action превращается в ask_for_clarification со списком в metadata.
        - Нет совпадений → ask_for_clarification с открытым вопросом.
        """
        if not profile.trello_board_id:
            raise BoardNotLinkedError()
        match_text = (action.match_text or user_text or "").strip()
        if not match_text:
            return (
                action.model_copy(
                    update={
                        "action_type": "ask_for_clarification",
                        "question": "Что именно вы выполнили? Напишите коротко (например, «помыл окна»).",
                        "metadata": {**(action.metadata or {}), "after_clarification": "complete_task"},
                    }
                ),
                None,
            )

        lists_raw = await self.trello_client.list_lists(profile.trello_board_id)
        cards = await self.trello_client.list_board_cards_with_checklists(profile.trello_board_id)
        cards = drop_cards_in_archived_lists(cards, lists_raw)
        items = flatten_incomplete_items(cards)
        matches = best_matches(match_text, items)

        if not matches:
            return (
                action.model_copy(
                    update={
                        "action_type": "ask_for_clarification",
                        "question": (
                            f"Не нашёл активного пункта чеклиста, похожего на «{match_text}». "
                            "Уточните формулировку или пришлите ссылку на карточку."
                        ),
                        "metadata": {
                            **(action.metadata or {}),
                            "after_clarification": "complete_task",
                            "match_text": match_text,
                        },
                    }
                ),
                None,
            )

        unique = confident_unique(matches)
        if unique is not None:
            new_action = action.model_copy(
                update={
                    "action_type": "update_checklist_item",
                    "card_id": unique.card_id,
                    "checklist_id": unique.checklist_id,
                    "check_item_id": unique.check_item_id,
                    "check_item_complete": True,
                    "metadata": {**(action.metadata or {}), "matched_item_name": unique.item_name},
                }
            )
            return (
                new_action,
                f"Отметил «{unique.item_name}» в карточке «{unique.card_name}» как выполненный.",
            )

        candidates = [
            {
                "card_id": ref.card_id,
                "card_name": ref.card_name,
                "checklist_id": ref.checklist_id,
                "checklist_name": ref.checklist_name,
                "check_item_id": ref.check_item_id,
                "item_name": ref.item_name,
                "score": s,
            }
            for ref, s in matches
        ]
        listing = "\n".join(
            f"{i + 1}) {c['card_name']} → {c['item_name']}" for i, c in enumerate(candidates)
        )
        question = (
            f"Не уверен, какой пункт отметить выполненным для «{match_text}». Выберите номер:\n{listing}"
        )
        return (
            action.model_copy(
                update={
                    "action_type": "ask_for_clarification",
                    "question": question,
                    "metadata": {
                        **(action.metadata or {}),
                        "after_clarification": "complete_task",
                        "match_text": match_text,
                        "candidates": candidates,
                    },
                }
            ),
            None,
        )

    async def _complete_all_incomplete_checklist_items(self, card_id: str) -> int:
        """Отмечает все незавершённые пункты чеклистов карточки как выполненные. Возвращает число обновлённых пунктов."""
        checklists = await self.trello_client.list_card_checklists(card_id)
        n = 0
        for cl in checklists:
            cid = str(cl.get("id") or "")
            if not cid:
                continue
            for ci in cl.get("checkItems") or []:
                if (ci.get("state") or "incomplete") == "complete":
                    continue
                iid = str(ci.get("id") or "")
                if not iid:
                    continue
                await self.trello_client.update_check_item_on_card(
                    card_id, cid, iid, complete=True,
                )
                n += 1
        return n

    async def _resolve_complete_card(
        self,
        profile: UserProfile,
        user_text: str,
        action: AgentAction,
    ) -> tuple[AgentAction, str | None]:
        """Ищем карточку по названию и переводим в колонку «Завершённые».

        Если есть невыполненные пункты — спрашиваем подтверждение (force).
        Если несколько кандидатов — список номером.
        """
        if not profile.trello_board_id:
            raise BoardNotLinkedError()
        if not profile.trello_done_list_id:
            raise OrchestratorValidationError(
                "У профиля не задан список «Завершённые» (done) на доске Trello. Привяжите доску заново через /link.",
            )
        match_text = (action.match_text or user_text or "").strip()
        if not match_text:
            return (
                action.model_copy(
                    update={
                        "action_type": "ask_for_clarification",
                        "question": "Какую карточку перевести в «Завершённые»? Назовите её коротко.",
                        "metadata": {**(action.metadata or {}), "after_clarification": "complete_card"},
                    }
                ),
                None,
            )

        lists_raw = await self.trello_client.list_lists(profile.trello_board_id)
        payload = await self.trello_client.list_board_cards_with_checklists(profile.trello_board_id)
        payload = drop_cards_in_archived_lists(payload, lists_raw)
        cards = cards_from_payload(payload)
        # Не пытаемся завершить уже находящиеся в Done.
        cards = [c for c in cards if c.list_id != profile.trello_done_list_id]

        hits = await self._search_cards_hybrid(profile, match_text, cards)
        if not hits:
            return (
                action.model_copy(
                    update={
                        "action_type": "ask_for_clarification",
                        "question": (
                            f"Не нашёл активной карточки, похожей на «{match_text}». "
                            "Уточните название или пришлите ссылку на карточку."
                        ),
                        "metadata": {
                            **(action.metadata or {}),
                            "after_clarification": "complete_card",
                            "match_text": match_text,
                        },
                    }
                ),
                None,
            )

        winner = confident_unique_hit(hits)
        if winner is not None:
            ref = next((c for c in cards if c.card_id == winner.card_id), None)
            if ref is not None:
                return self._build_complete_card_followup(profile, action, ref)

        candidates = [
            {
                "card_id": h.card_id,
                "card_name": h.card_name,
                "list_id": h.list_id,
                "incomplete_items": list(h.incomplete_items),
                "total_items": h.total_items,
                "score": h.score,
                "reason": h.reason,
            }
            for h in hits
        ]
        listing = "\n".join(f"{i + 1}) {c['card_name']}" for i, c in enumerate(candidates))
        question = f"Нашёл несколько подходящих карточек для «{match_text}». Выберите номер:\n{listing}"
        return (
            action.model_copy(
                update={
                    "action_type": "ask_for_clarification",
                    "question": question,
                    "metadata": {
                        **(action.metadata or {}),
                        "after_clarification": "complete_card",
                        "match_text": match_text,
                        "candidates": candidates,
                    },
                }
            ),
            None,
        )

    async def _search_cards_hybrid(
        self,
        profile: UserProfile,
        query: str,
        cards: list[CardRef],
    ) -> list[CardSearchHit]:
        """Гибридный поиск: fuzzy + LLM-семантика. Объединяем по card_id.

        Если fuzzy уже даёт уверенный единственный матч с сильным score,
        пропускаем LLM-вызов (экономия latency и токенов).
        """
        if not cards or not query:
            return []

        fuzzy = best_card_matches(query, cards)
        # Если fuzzy уверенно нашёл одну карточку с высоким score — не тратим LLM.
        if fuzzy:
            top_ref, top_score = fuzzy[0]
            second_score = fuzzy[1][1] if len(fuzzy) > 1 else 0.0
            if top_score >= 0.7 and (top_score - second_score) >= 0.18:
                logger.info(
                    "card search: fuzzy confident, skip LLM. winner=%r score=%.2f",
                    top_ref.card_name,
                    top_score,
                )
                return [
                    CardSearchHit(
                        card_id=top_ref.card_id,
                        card_name=top_ref.card_name,
                        score=min(1.0, top_score),
                        reason="по названию",
                        incomplete_items=top_ref.incomplete_items,
                        total_items=top_ref.total_items,
                        list_id=top_ref.list_id,
                    )
                ]

        # Иначе — добиваем LLM-семантикой.
        cards_payload = format_cards_for_search(cards)
        try:
            semantic = await self.agent_service.search_cards_semantic(
                query, cards_payload, persona=profile.persona.value,
            )
        except Exception:  # noqa: BLE001
            logger.exception("card_search_semantic failed; используем только fuzzy")
            semantic = []
        merged = merge_search_results(fuzzy, semantic, cards)
        logger.info(
            "card search hybrid: fuzzy=%d semantic=%d merged=%d query=%r",
            len(fuzzy),
            len(semantic),
            len(merged),
            query,
        )
        return merged

    def _build_complete_card_followup(
        self,
        profile: UserProfile,
        action: AgentAction,
        card: CardRef,
    ) -> tuple[AgentAction, str | None]:
        """Уверенный кандидат: либо подтверждение из-за невыполненных пунктов, либо move_card."""
        if card.incomplete_items:
            preview = "; ".join(card.incomplete_items[:5])
            extra = "" if len(card.incomplete_items) <= 5 else f" и ещё {len(card.incomplete_items) - 5}"
            return (
                action.model_copy(
                    update={
                        "action_type": "ask_for_clarification",
                        "question": (
                            f"В карточке «{card.card_name}» есть невыполненные пункты: {preview}{extra}. "
                            "Всё равно перевести в «Завершённые»? Ответьте «да» или «нет»."
                        ),
                        "metadata": {
                            **(action.metadata or {}),
                            "after_clarification": "complete_card_force",
                            "card_id": card.card_id,
                            "card_name": card.card_name,
                            "remaining_items": list(card.incomplete_items),
                        },
                    }
                ),
                None,
            )
        new_action = action.model_copy(
            update={
                "action_type": "move_card",
                "card_id": card.card_id,
                "list_name": "done",
                "metadata": {**(action.metadata or {}), "matched_card_name": card.card_name},
            }
        )
        return new_action, f"Перевёл карточку «{card.card_name}» в «Завершённые»."

    async def _finish_turn(self, profile: UserProfile, user_text: str, result: AgentResult) -> OrchestratorResult:
        action = result.action
        logger.info(
            "orchestrator: incoming action_type=%s reasoning=%r user_text=%r",
            action.action_type,
            action.reasoning,
            user_text,
        )
        action = self._apply_due_fallback(user_text, action)
        action = self._ensure_create_card_complete(user_text, action)

        success_text: str | None = None
        if action.action_type == "create_checklist_item" and action.checklist_item and not action.card_id:
            action, st = await self._resolve_add_checklist_card(profile, user_text, action)
            if st:
                success_text = st

        action = self._ensure_checklist_has_card_or_ask(user_text, action)

        if action.action_type == "complete_task_from_text":
            action, success_text = await self._resolve_complete_task(profile, user_text, action)
        elif action.action_type == "complete_card_by_text":
            action, success_text = await self._resolve_complete_card(profile, user_text, action)

        if action.action_type == "ask_for_clarification" and action.question:
            await self.clarification_repo.upsert(
                profile.telegram_user_id,
                action.question,
                action.model_dump_json(),
            )
            return OrchestratorResult(action.question)

        if action.action_type == "set_persona":
            persona = action.metadata.get("persona")
            if persona in {"elon", "zen", "mom"}:
                profile.persona = Persona(persona)
                return OrchestratorResult(f"Режим ассистента переключен на: {persona}.")

        if action.action_type == "none":
            return OrchestratorResult(result.response_text)

        await self._execute_action(profile, action)
        return OrchestratorResult(success_text or result.response_text)

    @staticmethod
    def _pick_complete_task_candidate(
        candidates: list[dict[str, object]],
        reply: str,
    ) -> dict[str, object] | None:
        """Парсим выбор пользователя: «1», «второй», «помыть окна» → один кандидат."""
        if not candidates:
            return None
        s = (reply or "").strip().lower()
        if not s:
            return None
        idx = _extract_index_from_reply(s, len(candidates))
        if idx is not None:
            return candidates[idx - 1]
        ranked = sorted(
            candidates,
            key=lambda c: SequenceMatcher(None, s, str(c.get("item_name", "")).lower()).ratio(),
            reverse=True,
        )
        if ranked and SequenceMatcher(None, s, str(ranked[0].get("item_name", "")).lower()).ratio() >= 0.55:
            return ranked[0]
        return None

    @staticmethod
    def _pick_complete_card_candidate(
        candidates: list[dict[str, object]],
        reply: str,
    ) -> dict[str, object] | None:
        """Парсим выбор: «1», «второй», «один», «под номером один» → одна карточка."""
        if not candidates:
            return None
        s = (reply or "").strip().lower()
        if not s:
            return None
        idx = _extract_index_from_reply(s, len(candidates))
        if idx is not None:
            return candidates[idx - 1]
        ranked = sorted(
            candidates,
            key=lambda c: SequenceMatcher(None, s, str(c.get("card_name", "")).lower()).ratio(),
            reverse=True,
        )
        if ranked and SequenceMatcher(None, s, str(ranked[0].get("card_name", "")).lower()).ratio() >= 0.55:
            return ranked[0]
        return None

    @staticmethod
    def _looks_like_new_command(text: str) -> bool:
        """Эвристика: реплика выглядит как НОВАЯ команда, а не ответ на уточнение.

        Используем когда у пользователя залип pending clarification и нужно
        понять, продолжает ли он отвечать на старый вопрос или дал новую команду.
        """
        s = (text or "").strip().lower()
        if not s:
            return False
        # Короткие ответы — это с большой вероятностью реакция на уточнение.
        if len(s) <= 3:
            return False
        # Явный выбор пункта/номера — это ответ на уточнение, а не новая команда.
        if re.search(r"\bномер\w*\s+\w+", s):
            return False
        if _extract_index_from_reply(s, 99) is not None:
            return False
        markers = (
            "постав", "создай", "запланир", "напомни", "не забы",
            "закрой", "удали", "сотри", "перенес", "обнови", "переименуй",
            "новая задач", "новую задач", "новое задание",
            "сделай задач",
        )
        return any(m in s for m in markers)

    @staticmethod
    def _is_explicit_new_command(text: str, *, expected_intents: set[str]) -> bool:
        """True если ответ явно выглядит как новая команда, а не продолжение уточнения."""
        if TaskOrchestrator._looks_like_new_command(text):
            return True
        ruled = quick_classify_intent(text)
        if ruled is None:
            return False
        return ruled not in expected_intents

    @staticmethod
    def _parse_yes_no(reply: str) -> bool | None:
        """Парсим да/нет в свободной форме."""
        raw = (reply or "").strip()
        raw = raw.replace("\ufeff", "").replace("\u200b", "").strip()
        s = unicodedata.normalize("NFC", raw).lower()
        if not s:
            return None
        # Whisper иногда даёт латиницу "Da" вместо «да»
        yes_tokens = {
            "да",
            "ага",
            "угу",
            "конечно",
            "yes",
            "y",
            "da",
            "ок",
            "окей",
            "go",
            "давай",
            "переводи",
            "+",
        }
        no_tokens = {"нет", "не", "нельзя", "no", "n", "net", "стоп", "отмена", "-"}
        first = re.split(r"[\s,.!?;:]+", s)[0]
        if first in yes_tokens:
            return True
        if first in no_tokens:
            return False
        if any(t in s for t in ("всё равно перевед", "все равно перевед", "переводи", "закрой всё равно")):
            return True
        if any(t in s for t in ("не перевод", "не закры", "сначала отмеч")):
            return False
        return None

    @staticmethod
    def _implies_complete_card_force_yes(reply: str) -> bool:
        """Ответ не начинается с «да», но по смыслу подтверждает закрытие / сообщает о сделанном пункте (голос)."""
        s = unicodedata.normalize("NFC", (reply or "").strip().lower())
        if not s:
            return False
        # Уже выполнили то, что висело в чеклисте (частый голосовой ответ).
        if "лобов" in s and "помы" in s:
            return True
        if "стекло" in s and "помы" in s:
            return True
        if any(t in s for t in ("всё равно закрыв", "все равно закрыв", "закрывай", "переводи в заверш")):
            return True
        return False

    async def process_text(self, profile: UserProfile, text: str) -> OrchestratorResult:
        pending = await self.clarification_repo.get(profile.telegram_user_id)
        if pending:
            await self.clarification_repo.clear(profile.telegram_user_id)
            try:
                draft = json.loads(pending.draft_action_json)
            except (ValueError, TypeError):
                draft = {}
            meta = draft.get("metadata") or {}
            after = meta.get("after_clarification")

            if after == "create_checklist_item":
                if self._is_explicit_new_command(text, expected_intents={"add_checklist_items"}):
                    logger.info(
                        "create_checklist_item pending dropped, routing as fresh input. text=%r",
                        text,
                    )
                    result = await self.agent_service.infer_action(
                        text=text, persona=profile.persona.value,
                    )
                    return await self._finish_turn(profile, text, result)
                resume = self._checklist_resume_prompt(meta, text)
                result = await self.agent_service.infer_action(
                    text=resume,
                    persona=profile.persona.value,
                    forced_intent="add_checklist_items",
                )
                return await self._finish_turn(profile, resume, result)

            if after == "add_checklist_pick_card":
                candidates = list(meta.get("candidates") or [])
                if candidates and not self._looks_like_new_command(text):
                    chosen = self._pick_complete_card_candidate(candidates, text)
                    if chosen is not None:
                        action = AgentAction(
                            action_type="create_checklist_item",
                            card_id=str(chosen.get("card_id")),
                            checklist_name=str(meta.get("pending_checklist_name") or "Шаги"),
                            checklist_item=str(meta.get("pending_checklist_item") or ""),
                        )
                        await self._execute_action(profile, action)
                        return OrchestratorResult(
                            f"Добавил пункты в карточку «{chosen.get('card_name')}».",
                        )
                logger.info(
                    "add_checklist_pick_card pending dropped, routing as fresh input. text=%r",
                    text,
                )
                result = await self.agent_service.infer_action(text=text, persona=profile.persona.value)
                return await self._finish_turn(profile, text, result)

            if after == "create_card":
                if self._is_explicit_new_command(text, expected_intents={"create_card"}):
                    logger.info(
                        "create_card pending dropped, routing as fresh input. text=%r",
                        text,
                    )
                    result = await self.agent_service.infer_action(
                        text=text, persona=profile.persona.value,
                    )
                    return await self._finish_turn(profile, text, result)
                resume = self._create_card_resume_prompt(meta, text)
                result = await self.agent_service.infer_action(
                    text=resume,
                    persona=profile.persona.value,
                    forced_intent="create_card",
                )
                return await self._finish_turn(profile, resume, result)

            if after == "complete_task":
                candidates = list(meta.get("candidates") or [])
                # Если в pending есть кандидаты И пользователь похож на «выбрал номер/имя» —
                # пытаемся резолвить. Иначе считаем, что это новая команда.
                if candidates and not self._looks_like_new_command(text):
                    chosen = self._pick_complete_task_candidate(candidates, text)
                    if chosen is not None:
                        action = AgentAction(
                            action_type="update_checklist_item",
                            card_id=str(chosen.get("card_id")),
                            checklist_id=str(chosen.get("checklist_id")),
                            check_item_id=str(chosen.get("check_item_id")),
                            check_item_complete=True,
                        )
                        await self._execute_action(profile, action)
                        return OrchestratorResult(
                            f"Отметил «{chosen.get('item_name')}» в карточке "
                            f"«{chosen.get('card_name')}» как выполненный.",
                        )
                # Зомби-pending: «не нашёл пункт» из прошлой попытки. Отпускаем в роутер.
                logger.info(
                    "complete_task pending dropped, routing as fresh input. text=%r candidates=%d",
                    text,
                    len(candidates),
                )
                result = await self.agent_service.infer_action(text=text, persona=profile.persona.value)
                return await self._finish_turn(profile, text, result)

            if after == "complete_card":
                candidates = list(meta.get("candidates") or [])
                if candidates and not self._looks_like_new_command(text):
                    chosen = self._pick_complete_card_candidate(candidates, text)
                    if chosen is not None:
                        card_ref = CardRef(
                            card_id=str(chosen.get("card_id")),
                            card_name=str(chosen.get("card_name")),
                            list_id=chosen.get("list_id"),
                            short_url=None,
                            incomplete_items=tuple(str(x) for x in (chosen.get("incomplete_items") or [])),
                            total_items=int(chosen.get("total_items") or 0),
                        )
                        placeholder = AgentAction(action_type="complete_card_by_text")
                        action, success_text = self._build_complete_card_followup(
                            profile, placeholder, card_ref,
                        )
                        if action.action_type == "ask_for_clarification" and action.question:
                            await self.clarification_repo.upsert(
                                telegram_user_id=profile.telegram_user_id,
                                question=action.question,
                                draft_action_json=action.model_dump_json(),
                            )
                            return OrchestratorResult(action.question)
                        await self._execute_action(profile, action)
                        return OrchestratorResult(success_text or "Готово.")
                logger.info(
                    "complete_card pending dropped, routing as fresh input. text=%r candidates=%d",
                    text,
                    len(candidates),
                )
                result = await self.agent_service.infer_action(text=text, persona=profile.persona.value)
                return await self._finish_turn(profile, text, result)

            if after == "complete_card_force":
                ans = self._parse_yes_no(text)
                if ans is None:
                    ans = True if self._implies_complete_card_force_yes(text) else None
                card_id = str(meta.get("card_id") or "")
                card_name = str(meta.get("card_name") or "")
                if ans is True and card_id:
                    n_done = await self._complete_all_incomplete_checklist_items(card_id)
                    action = AgentAction(action_type="move_card", card_id=card_id, list_name="done")
                    await self._execute_action(profile, action)
                    if n_done:
                        return OrchestratorResult(
                            f"Отметил выполненными пунктов чеклиста: {n_done}. "
                            f"Перевёл карточку «{card_name}» в «Завершённые».",
                        )
                    return OrchestratorResult(f"Перевёл карточку «{card_name}» в «Завершённые».")
                if ans is False:
                    return OrchestratorResult(
                        "Хорошо, не переводил. Сначала отметьте оставшиеся пункты выполненными "
                        "(скажите, что именно сделали), и тогда я закрою карточку.",
                    )
                # Если новый ввод похож на полноценную команду — выходим из force-режима.
                if self._looks_like_new_command(text):
                    logger.info(
                        "complete_card_force pending dropped, routing as fresh input. text=%r",
                        text,
                    )
                    result = await self.agent_service.infer_action(
                        text=text, persona=profile.persona.value,
                    )
                    return await self._finish_turn(profile, text, result)
                # Иначе — переспрашиваем и сохраняем контекст.
                question = (
                    f"Не понял ответ. Перевести карточку «{card_name}» в «Завершённые», "
                    "несмотря на невыполненные пункты? Ответьте «да» или «нет»."
                )
                pending_action = AgentAction(
                    action_type="ask_for_clarification",
                    question=question,
                    metadata={
                        "after_clarification": "complete_card_force",
                        "card_id": card_id,
                        "card_name": card_name,
                        "remaining_items": list(meta.get("remaining_items") or []),
                    },
                )
                await self.clarification_repo.upsert(
                    telegram_user_id=profile.telegram_user_id,
                    question=question,
                    draft_action_json=pending_action.model_dump_json(),
                )
                return OrchestratorResult(question)

            # Совместимость: старый путь без after_clarification — пробуем выполнить как есть,
            # но защищаемся от silent failure для самого ask_for_clarification.
            if draft.get("action_type") in (None, "ask_for_clarification"):
                result = await self.agent_service.infer_action(text=text, persona=profile.persona.value)
                return await self._finish_turn(profile, text, result)

            draft["metadata"] = {**meta, "clarification_answer": text}
            action = AgentAction.model_validate(draft)
            await self._execute_action(profile, action)
            return OrchestratorResult("Спасибо за уточнение. Готово.")

        result = await self.agent_service.infer_action(text=text, persona=profile.persona.value)
        return await self._finish_turn(profile, text, result)

    async def _execute_action(self, profile: UserProfile, action: AgentAction) -> None:
        if not profile.trello_board_id:
            raise BoardNotLinkedError()
        lists_map = {
            "inbox": profile.trello_inbox_list_id,
            "doing": profile.trello_doing_list_id,
            "done": profile.trello_done_list_id,
        }
        if action.action_type == "create_card":
            list_key = (action.list_name or "inbox").lower()
            list_id = lists_map.get(list_key) or profile.trello_inbox_list_id
            if not list_id or not action.card_name:
                raise OrchestratorValidationError("Недостаточно данных для создания карточки (название / колонка).")
            card = await self.trello_client.create_card(
                list_id,
                action.card_name,
                action.card_description or "",
                due=action.due,
                start=action.start,
                due_complete=action.due_complete,
                position=action.position,
            )
            # Use-case 2: если в этом же действии есть пункты чеклиста — добавляем их.
            items = _split_checklist_items(action.checklist_item)
            if items and isinstance(card, dict) and card.get("id"):
                checklist = await self.trello_client.add_checklist(
                    card["id"],
                    action.checklist_name or "Шаги",
                )
                for item in items:
                    await self.trello_client.add_check_item(checklist["id"], item)
        elif action.action_type == "update_card":
            if not action.card_id:
                raise OrchestratorValidationError("Нужен card_id для обновления карточки.")
            has_patch = (
                action.card_name is not None
                or action.card_description is not None
                or action.due is not None
                or action.start is not None
                or action.due_complete is not None
                or action.closed is not None
            )
            if not has_patch:
                raise OrchestratorValidationError(
                    "Укажите, что изменить: название, описание, срок due/start, выполнение due_complete или архив closed.",
                )
            await self.trello_client.update_card(
                action.card_id,
                action.card_name,
                action.card_description,
                due=action.due,
                start=action.start,
                due_complete=action.due_complete,
                closed=action.closed,
            )
        elif action.action_type == "delete_card":
            if not action.card_id:
                raise OrchestratorValidationError("Нужен card_id для удаления карточки.")
            await self.trello_client.delete_card(action.card_id)
        elif action.action_type == "move_card":
            if not action.card_id:
                raise OrchestratorValidationError("Нужен card_id для перемещения карточки.")
            target_id = lists_map.get((action.list_name or "").lower())
            if not target_id:
                raise OrchestratorValidationError("Не удалось определить целевой статус (колонку).")
            await self.trello_client.move_card(action.card_id, target_id)
        elif action.action_type == "create_checklist_item":
            if not action.card_id or not action.checklist_item:
                raise OrchestratorValidationError("Нужны card_id и текст пункта чеклиста.")
            parts = _split_checklist_items(action.checklist_item)
            if not parts:
                raise OrchestratorValidationError("Пустой пункт чеклиста.")
            checklist = await self.trello_client.add_checklist(action.card_id, action.checklist_name or "Checklist")
            for part in parts:
                await self.trello_client.add_check_item(checklist["id"], part)
        elif action.action_type == "attach_file":
            if not action.card_id or not action.file_url:
                raise OrchestratorValidationError("Нужны card_id и ссылка на файл.")
            await self.trello_client.attach_file_by_url(action.card_id, action.file_url)
        elif action.action_type == "add_comment":
            if not action.card_id or not action.comment_text:
                raise OrchestratorValidationError("Нужны card_id и текст комментария (comment_text).")
            await self.trello_client.add_card_comment(action.card_id, action.comment_text)
        elif action.action_type == "add_card_label":
            if not action.card_id or not action.label_id:
                raise OrchestratorValidationError("Нужны card_id и label_id (id метки на доске в Trello).")
            await self.trello_client.add_label_to_card(action.card_id, action.label_id.strip())
        elif action.action_type == "remove_card_label":
            if not action.card_id or not action.label_id:
                raise OrchestratorValidationError("Нужны card_id и label_id метки для снятия.")
            await self.trello_client.remove_label_from_card(action.card_id, action.label_id.strip())
        elif action.action_type == "add_card_member":
            if not action.card_id or not action.member_id:
                raise OrchestratorValidationError("Нужны card_id и member_id (id участника Trello).")
            await self.trello_client.add_member_to_card(action.card_id, action.member_id.strip())
        elif action.action_type == "remove_card_member":
            if not action.card_id or not action.member_id:
                raise OrchestratorValidationError("Нужны card_id и member_id для снятия с карточки.")
            await self.trello_client.remove_member_from_card(action.card_id, action.member_id.strip())
        elif action.action_type == "update_checklist_item":
            if not action.card_id or not action.checklist_id or not action.check_item_id:
                raise OrchestratorValidationError(
                    "Нужны card_id, checklist_id и check_item_id — их можно взять из JSON карточки в Trello API или из URL.",
                )
            has_upd = action.check_item_complete is not None or action.check_item_new_name is not None
            if not has_upd:
                raise OrchestratorValidationError(
                    "Укажите check_item_complete (true/false) и/или check_item_new_name для переименования пункта.",
                )
            await self.trello_client.update_check_item_on_card(
                action.card_id,
                action.checklist_id,
                action.check_item_id,
                complete=action.check_item_complete,
                name=action.check_item_new_name,
            )
