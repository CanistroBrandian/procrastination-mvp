from __future__ import annotations

import json
import logging
import re
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
from app.services.trello_board_filters import drop_cards_in_archived_lists


@dataclass
class OrchestratorResult:
    text: str


def _split_checklist_items(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in re.split(r"[\n;|]+", raw) if p.strip()]


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
        return action.model_copy(
            update={
                "action_type": "ask_for_clarification",
                "question": (
                    "Чтобы добавить пункты в чеклист, нужна карточка в Trello. Пришлите ссылку на карточку "
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
            "Верни JSON: если из текста однозначно следует card_id (24 символа или из ссылки trello.com/c/shortlink) — "
            "create_checklist_item с этим card_id и тем же checklist_item. Если id не ясен — ask_for_clarification."
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
        action = self._ensure_checklist_has_card_or_ask(user_text, action)

        success_text: str | None = None
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
        m = re.search(r"\b(\d{1,2})\b", s)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        # Попробуем по подстроке имени пункта.
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
        """Парсим выбор: «1», «2», «обслуживание машины» → одна карточка."""
        if not candidates:
            return None
        s = (reply or "").strip().lower()
        if not s:
            return None
        m = re.search(r"\b(\d{1,2})\b", s)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= len(candidates):
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
        markers = (
            "постав", "задач", "создай", "запланир", "напомни", "не забы",
            "закрой", "удали", "сотри", "перенес", "обнови", "переименуй",
            "добавь", "новая", "новую", "новое", "сделай задач",
        )
        return any(m in s for m in markers)

    @staticmethod
    def _parse_yes_no(reply: str) -> bool | None:
        """Парсим да/нет в свободной форме."""
        s = (reply or "").strip().lower()
        if not s:
            return None
        yes_tokens = {"да", "ага", "угу", "конечно", "yes", "y", "ок", "окей", "go", "давай", "переводи", "+"}
        no_tokens = {"нет", "не", "нельзя", "no", "n", "стоп", "отмена", "-"}
        first = re.split(r"[\s,.!?]+", s)[0]
        if first in yes_tokens:
            return True
        if first in no_tokens:
            return False
        if any(t in s for t in ("всё равно перевед", "все равно перевед", "переводи", "закрой всё равно")):
            return True
        if any(t in s for t in ("не перевод", "не закры", "сначала отмеч")):
            return False
        return None

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
                resume = self._checklist_resume_prompt(meta, text)
                result = await self.agent_service.infer_action(
                    text=resume,
                    persona=profile.persona.value,
                    forced_intent="add_checklist_items",
                )
                return await self._finish_turn(profile, resume, result)

            if after == "create_card":
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
                card_id = str(meta.get("card_id") or "")
                card_name = str(meta.get("card_name") or "")
                if ans is True and card_id:
                    action = AgentAction(action_type="move_card", card_id=card_id, list_name="done")
                    await self._execute_action(profile, action)
                    return OrchestratorResult(
                        f"Перевёл карточку «{card_name}» в «Завершённые», часть пунктов осталась невыполненной.",
                    )
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
