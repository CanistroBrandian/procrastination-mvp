"""Двухшаговый LLM-агент: классификатор намерений + узкий извлекатель полей.

Архитектура:
  1) IntentRouter — крошечный промпт. Только выбирает один из ~17 интентов.
     Чёткие маркеры по времени глаголов и роли (карточка vs пункт чеклиста).
  2) IntentExtractor — узкий промпт под выбранный intent. Заполняет только
     релевантные поля. Не знает про другие интенты, поэтому не путается.
  3) AgentService.infer_action возвращает прежний AgentResult — оркестратор
     не нуждается в правках.

Преимущества по сравнению со старым одношаговым промптом:
- Каждый промпт компактен и однозначен → выше точность на слабых моделях.
- Логика выбора интента отделена от извлечения данных, проще отлаживать.
- Можно подменять/добавлять интенты, не разрастая один гигантский prompt.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from pydantic import ValidationError

from app.core.errors import OpenRouterLLMError
from app.schemas.actions import AgentAction, AgentResult
from app.schemas.intents import VALID_INTENTS, IntentDecision
from app.services.card_search import CardSearchHit
from app.services.date_infer import calendar_context_for_prompt
from app.services.intent_rules import quick_classify_intent


logger = logging.getLogger(__name__)


PERSONA_PROMPTS = {
    "elon": (
        "Ты строгий мотиватор. Пиши коротко, энергично, без оскорблений. "
        "Фокусируйся на цене бездействия и срочности первого шага."
    ),
    "zen": (
        "Ты спокойный дзен-наставник. Помогаешь найти препятствие и выбрать маленький следующий шаг."
    ),
    "mom": (
        "Ты поддерживающий и теплый помощник. Веришь в пользователя и даешь конструктивную поддержку."
    ),
}


# ============================================================================
# 1) ROUTER PROMPT — только классификация intent
# ============================================================================

ROUTER_PROMPT = """\
Ты — классификатор намерений для задачника на базе Trello.
На вход получаешь реплику пользователя (русский, разговорный, может быть транскрибировано из голоса).
Твоя ЕДИНСТВЕННАЯ задача — определить, какое действие хочет совершить пользователь.

Верни СТРОГО JSON: {"intent":"<один из списка>","reasoning":"<2-3 фразы>","response_text":"<краткий ответ пользователю>"}

Интенты (выбери ОДИН):

1) create_card — поставить НОВУЮ задачу.
   ЖЁСТКИЙ ПРИОРИТЕТ: если в реплике есть «поставим задачу», «поставь задачу», «зададим задачу»,
   «запланируй задачу», «давай задачу», «новая задача», «напомни», «не забыть» — это ВСЕГДА create_card,
   ДАЖЕ если рядом есть глагол «сделать» / «делать» / «помыть» в инфинитиве (это тогда не «выполнено», а описание сути новой задачи).
   Маркеры: будущее время + новая активность.
   Примеры:
     "Слушай, давай на завтра поставим задачу делать уроки" → create_card (card_name="Сделать уроки").
     "Давай назовем поставим задачу, надо короче уроки мне сделать и там математика, русский язык и географию"
        → create_card (card_name="Сделать уроки", checklist=["Математика","Русский язык","География"]).
     "Завтра зададим задачу посадить дерево, нужно купить грунт и лопату" → create_card.
     "Напомни купить хлеб" → create_card.

2) complete_task_from_text — отметить ОДИН ПУНКТ внутри уже существующей задачи как ВЫПОЛНЕННЫЙ.
   Маркеры: "я сделал", "я помыл", "закончил Х", "пропылесосил", "готово: Y", "помыл окна".
   КРИТИЧНО: глагол в ПРОШЕДШЕМ времени, описывает мелкую подзадачу, не всю задачу целиком.
   Примеры:
     "Пропылесосил в машине" → complete_task_from_text.
     "Помыл окна" → complete_task_from_text.
     "Закончил презентацию для проекта А" → complete_task_from_text.

3) complete_card_by_text — закрыть/завершить ВСЮ карточку (основную задачу).
   Маркеры: "закрой карточку Х", "переведи задачу Х в завершённые", "Х готова, закрывай",
   "обслуживание машины — готово", "машину обслужили, можно закрывать эту задачу".
   КРИТИЧНО: говорят о ЦЕЛОЙ задаче, а не о пункте внутри неё.
   Примеры:
     "Машину мы обслужили, короче можно закрывать эту задачу" → complete_card_by_text.
     "Карточка обслуживание машины — переведи в завершённые" → complete_card_by_text.

4) update_card — изменить поле существующей карточки (срок, название, описание, дата начала).
   Маркеры: "перенеси на пятницу", "переименуй Х в Y", "поставь срок 10 мая на Х", "сделай Х на завтра".

5) delete_card — удалить карточку. Маркеры: "удали", "сотри", "убери карточку".

6) move_card — переместить в другую колонку без завершения. Маркеры: "перенеси в doing", "в работу".

7) add_checklist_items — добавить пункты в чеклист УЖЕ существующей карточки.
   Маркеры: "добавь к карточке Х пункт Y", "в чеклист Х добавь Y" (есть отсылка к существующей карточке).

8) update_checklist_item — переименовать конкретный пункт чеклиста.

9) add_comment — добавить комментарий к карточке. Маркеры: "прокомментируй", "добавь коммент".

10-13) add_card_label / remove_card_label / add_card_member / remove_card_member.

14) attach_file — прикрепить файл/ссылку к карточке.

15) set_persona — переключить режим: "будь как мама/илон/дзен", "переключись на zen".

16) link_board — пользователь привязывает доску Trello (присылает ссылку https://trello.com/b/...).

17) chitchat — нет конкретного действия. Приветствие, благодарность, общий вопрос.

ПРАВИЛА:
- При сомнении между create_card и complete_task_from_text смотри на ВРЕМЯ ГЛАГОЛА:
  будущее/инфинитив = create_card; прошедшее = complete_task_from_text.
- При сомнении между complete_task_from_text (пункт) и complete_card_by_text (вся карточка):
  если упомянуто название карточки целиком и слова "закрыть/завершить/готова" — это complete_card_by_text.
- response_text — короткая (1 фраза) реплика пользователю с подтверждением.
"""


# ============================================================================
# 2) EXTRACTOR PROMPTS — узкие промпты под каждый intent
# ============================================================================

EXTRACTOR_PROMPTS: dict[str, str] = {
    "create_card": """\
Извлеки поля для создания НОВОЙ карточки в Trello.
Верни JSON: {"action_type":"create_card","card_name":"...","due":"...","start":null,"position":null,"checklist_name":"...","checklist_item":[...],"card_description":null,"response_text":"..."}.
Правила:
- card_name — короткое ёмкое название задачи (без времени и без вспомогательных слов "не забыть/нужно/давай поставим").
- due — ISO 8601 (например 2026-05-15T09:00:00.000Z). Если нет ОДНОЗНАЧНОГО срока — поставь null (сервер уточнит).
- Если пользователь перечисляет подзадачи в той же фразе — checklist_item это МАССИВ строк (короткие пункты), checklist_name = "Шаги".
- Никогда не выдумывай дату.
""",
    "update_card": """\
Извлеки поля для обновления существующей карточки.
Верни JSON: {"action_type":"update_card","card_id":null,"card_name":null,"card_description":null,"due":null,"start":null,"due_complete":null,"closed":null,"response_text":"..."}.
Если в реплике явно указано, какое поле менять — заполни только его. Срок только в ISO 8601.
""",
    "delete_card": """\
Верни JSON: {"action_type":"delete_card","card_id":null,"card_name":null,"response_text":"..."}.
Если пользователь назвал карточку — занеси её в card_name (короткое название).
""",
    "move_card": """\
Верни JSON: {"action_type":"move_card","card_id":null,"card_name":null,"list_name":"inbox|doing|done","response_text":"..."}.
list_name — целевая колонка из {inbox, doing, done}.
""",
    "complete_card_by_text": """\
Извлеки название карточки, которую пользователь хочет ЗАКРЫТЬ/ПЕРЕВЕСТИ В ЗАВЕРШЁННЫЕ.
Верни JSON: {"action_type":"complete_card_by_text","match_text":"...","response_text":"..."}.
match_text — короткое название карточки БЕЗ слов "закрой/перевести/завершить/карточка/задача".
Пример: "Машину обслужили, можно закрывать эту задачу" → match_text = "обслуживание машины".
Пример: "Закрой карточку посадить дерево" → match_text = "посадить дерево".
Не подставляй card_id, сервер найдёт карточку по названию.
""",
    "add_checklist_items": """\
Извлеки пункты для добавления в чеклист УЖЕ существующей карточки.
Верни JSON: {"action_type":"create_checklist_item","card_id":null,"card_name":null,"checklist_name":"Шаги","checklist_item":"...","response_text":"..."}.
- checklist_item — строка или несколько пунктов через "; " / перевод строки (только то, что добавляем в чеклист).
- card_name — ТОЛЬКО короткое название самой карточки на доске. Никогда не помещай сюда предметы/пункты
  из checklist_item (например «английский», «география»). Если название дано в кавычках
  («Сделать уроки») — это название карточки, его можно продублировать в card_name или оставить null.
- Если есть ссылка trello.com/c/... или 24-символьный id — занеси в card_id.
- Если нет ни id, ни названия — card_id и card_name null; сервер извлечёт название из кавычек в тексте.
""",
    "complete_task_from_text": """\
Извлеки короткое описание того, что пользователь УЖЕ ВЫПОЛНИЛ (один пункт чеклиста).
Верни JSON: {"action_type":"complete_task_from_text","match_text":"...","response_text":"..."}.
match_text — без местоимений и вспомогательных слов: "я помыл лобовое стекло" → "помыл лобовое стекло".
Не выдумывай card_id и checklist_id — сервер найдёт пункт по тексту.
""",
    "update_checklist_item": """\
Верни JSON: {"action_type":"update_checklist_item","card_id":null,"checklist_id":null,"check_item_id":null,"check_item_complete":null,"check_item_new_name":null,"response_text":"..."}.
Заполни только указанные пользователем поля.
""",
    "add_comment": """\
Верни JSON: {"action_type":"add_comment","card_id":null,"card_name":null,"comment_text":"...","response_text":"..."}.
""",
    "add_card_label": """\
Верни JSON: {"action_type":"add_card_label","card_id":null,"label_id":null,"response_text":"..."}.
""",
    "remove_card_label": """\
Верни JSON: {"action_type":"remove_card_label","card_id":null,"label_id":null,"response_text":"..."}.
""",
    "add_card_member": """\
Верни JSON: {"action_type":"add_card_member","card_id":null,"member_id":null,"response_text":"..."}.
""",
    "remove_card_member": """\
Верни JSON: {"action_type":"remove_card_member","card_id":null,"member_id":null,"response_text":"..."}.
""",
    "attach_file": """\
Верни JSON: {"action_type":"attach_file","card_id":null,"card_name":null,"file_url":null,"response_text":"..."}.
""",
    "set_persona": """\
Верни JSON: {"action_type":"set_persona","metadata":{"persona":"mom|elon|zen"},"response_text":"..."}.
""",
    "link_board": """\
Верни JSON: {"action_type":"link_board","metadata":{"board_url":"https://trello.com/b/..."},"response_text":"..."}.
""",
}


# ============================================================================
# 3) CARD SEARCH AGENT — семантический поиск карточек по реплике
# ============================================================================

CARD_SEARCH_PROMPT = """\
Ты — семантический поисковик по карточкам Trello.

На вход получаешь JSON {"query": "...", "cards": [{"id":"...","name":"...","incomplete_items":[...],"total_items":N}]}.
- query — реплика пользователя в свободной форме (русский, разговорный).
- cards — открытые карточки на доске пользователя.

Задача: вернуть JSON {"hits":[{"card_id":"...","score":0.0..1.0,"reason":"коротко почему"}]}.

ПРАВИЛА:
- Включай ТОЛЬКО те карточки, к которым реплика реально относится.
- Используй ТОЛЬКО id из переданного списка. Никогда не выдумывай id.
- Сортируй по score убыванию.
- score >= 0.7 — точный/уверенный матч; 0.4..0.7 — вероятный (тематика совпадает); < 0.4 — НЕ возвращай.
- Если ничего не подходит — верни {"hits":[]}.

ПРИМЕРЫ:
1) query: "машину обслужили, можно закрывать"
   cards: [{"id":"abc","name":"Обслуживание машины","incomplete_items":["Помыть лобовое"]},{"id":"def","name":"Купить хлеб"}]
   ответ: {"hits":[{"card_id":"abc","score":0.92,"reason":"name прямо описывает обслуживание машины"}]}
2) query: "отчёт по проекту А готов"
   cards: [{"id":"a","name":"Отчёт по проекту А"},{"id":"b","name":"Отчёт по проекту Б"}]
   ответ: {"hits":[{"card_id":"a","score":0.95,"reason":"явное упоминание проекта А"}]}
3) query: "закрой что-то про машину"
   cards: [{"id":"abc","name":"Обслуживание машины"},{"id":"xyz","name":"Купить машину"}]
   ответ: {"hits":[{"card_id":"abc","score":0.6,"reason":"машина"},{"id":"xyz","score":0.55,"reason":"машина"}]}
"""


# ============================================================================
# Нормализаторы JSON-payload (защита от типичных «странностей» LLM)
# ============================================================================


def _normalize_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    rt = out.get("response_text")
    if rt is None or (isinstance(rt, str) and not rt.strip()):
        out["response_text"] = "Принято, работаем."
    elif not isinstance(rt, str):
        out["response_text"] = str(rt).strip() or "Принято, работаем."
    if out.get("metadata") is None:
        out["metadata"] = {}
    ci = out.get("checklist_item")
    if isinstance(ci, list):
        parts = [str(x).strip() for x in ci if x is not None and str(x).strip()]
        out["checklist_item"] = "; ".join(parts) if parts else None
    return out


def _normalize_router_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    intent = out.get("intent")
    if intent not in VALID_INTENTS:
        out["intent"] = "chitchat"
    if not isinstance(out.get("reasoning"), str):
        out["reasoning"] = ""
    rt = out.get("response_text")
    if rt is None or not isinstance(rt, str) or not rt.strip():
        out["response_text"] = "Принято, работаем."
    return out


def _render_intent_context_block(intent_context: str | None) -> str:
    if not intent_context:
        return ""
    trimmed = intent_context.strip()
    if not trimmed:
        return ""
    if len(trimmed) > 1800:
        trimmed = trimmed[-1800:]
    return (
        "\n\n[Intent History Context]\n"
        f"{trimmed}\n"
        "Use this context as a hint for disambiguation (create/update/complete). "
        "Never invent card ids and never execute older commands automatically."
    )


# ============================================================================
# Service
# ============================================================================


class AgentService:
    """LLM-агент. Двухшаговый: сначала classify_intent, потом extract_fields."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def infer_action(
        self,
        text: str,
        persona: str = "mom",
        *,
        forced_intent: str | None = None,
        intent_context: str | None = None,
    ) -> AgentResult:
        if forced_intent and forced_intent in VALID_INTENTS:
            decision = IntentDecision(
                intent=forced_intent,  # type: ignore[arg-type]
                reasoning="forced_by_orchestrator",
                response_text="Принято.",
            )
            logger.info("intent_router: FORCED intent=%s text=%r", forced_intent, text)
        else:
            # Шаг 0: дешёвый rule-based pre-router. Если ключевые маркеры
            # однозначно указывают на интент — не платим за LLM.
            rule_intent = quick_classify_intent(text)
            if rule_intent is not None:
                decision = IntentDecision(
                    intent=rule_intent,  # type: ignore[arg-type]
                    reasoning=f"rule_based:{rule_intent}",
                    response_text="Принято, работаем.",
                )
                logger.info(
                    "intent_router: RULE-BASED intent=%s text=%r",
                    rule_intent,
                    text,
                )
            else:
                # Шаг 1: LLM-роутер для всех неоднозначных случаев.
                decision = await self._classify_intent(text, persona, intent_context=intent_context)
                logger.info(
                    "intent_router: LLM intent=%s reasoning=%r text=%r",
                    decision.intent,
                    decision.reasoning,
                    text,
                )

        if decision.intent == "chitchat":
            return AgentResult(
                action=AgentAction(action_type="none", reasoning=decision.reasoning),
                response_text=decision.response_text,
            )

        action = await self._extract_fields(text, persona, decision, intent_context=intent_context)
        logger.info(
            "intent_extractor: action_type=%s card_name=%r match_text=%r due=%r",
            action.action_type,
            action.card_name,
            action.match_text,
            action.due,
        )
        return AgentResult(action=action, response_text=decision.response_text)

    # ------------------------------------------------- semantic card search
    async def search_cards_semantic(
        self,
        query: str,
        cards_payload: list[dict[str, Any]],
        persona: str = "mom",
    ) -> list[CardSearchHit]:
        """LLM-семантический поиск карточек по тексту.

        cards_payload — компактные сводки от `card_search.format_cards_for_search`.
        Возвращает список `CardSearchHit` (только валидные id, score >= 0.4).
        Этот метод НЕ обращается к Trello — список карточек подаётся снаружи.
        """
        if not query.strip() or not cards_payload:
            return []

        user_msg = json.dumps(
            {"query": query, "cards": cards_payload}, ensure_ascii=False,
        )
        system = (
            f"{PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS['mom'])}\n\n"
            f"{CARD_SEARCH_PROMPT}"
        )
        try:
            raw = await self._call_llm_json(system, user_msg, step="card_search")
        except OpenRouterLLMError as exc:
            logger.warning("card_search LLM failed: %s", exc)
            return []

        hits_raw = raw.get("hits") if isinstance(raw, dict) else None
        if not isinstance(hits_raw, list):
            logger.warning("card_search: payload без массива hits: %s", raw)
            return []

        valid_ids = {str(c.get("id")): c for c in cards_payload if c.get("id")}
        out: list[CardSearchHit] = []
        for h in hits_raw:
            if not isinstance(h, dict):
                continue
            cid = str(h.get("card_id") or "").strip()
            if cid not in valid_ids:
                continue
            try:
                score = float(h.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(1.0, score))
            if score < 0.4:
                continue
            card = valid_ids[cid]
            out.append(
                CardSearchHit(
                    card_id=cid,
                    card_name=str(card.get("name", "")),
                    score=score,
                    reason=str(h.get("reason") or ""),
                    incomplete_items=tuple(card.get("incomplete_items") or []),
                    total_items=int(card.get("total_items") or 0),
                    list_id=None,
                )
            )
        out.sort(key=lambda h: h.score, reverse=True)
        logger.info(
            "card_search: query=%r hits=%s",
            query,
            [(h.card_id, round(h.score, 2), h.card_name) for h in out],
        )
        return out

    # ------------------------------------------------------------------ step 1
    async def _classify_intent(
        self,
        text: str,
        persona: str,
        *,
        intent_context: str | None = None,
    ) -> IntentDecision:
        system = (
            f"{PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS['mom'])}\n"
            f"{calendar_context_for_prompt()}\n\n"
            f"{ROUTER_PROMPT}"
            f"{_render_intent_context_block(intent_context)}"
        )
        raw_payload = await self._call_llm_json(system, text, step="router")
        raw_payload = _normalize_router_payload(raw_payload)
        try:
            return IntentDecision.model_validate(raw_payload)
        except ValidationError as exc:
            logger.warning("intent_router validation failed: %s payload=%s", exc, raw_payload)
            return IntentDecision(intent="chitchat", reasoning="router_validation_failed")

    # ------------------------------------------------------------------ step 2
    async def _extract_fields(
        self,
        text: str,
        persona: str,
        decision: IntentDecision,
        *,
        intent_context: str | None = None,
    ) -> AgentAction:
        intent = decision.intent
        extractor_prompt = EXTRACTOR_PROMPTS.get(intent)
        if extractor_prompt is None:
            logger.warning("no extractor for intent=%s, returning none", intent)
            return AgentAction(action_type="none", reasoning=decision.reasoning)

        system = (
            f"{PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS['mom'])}\n"
            f"{calendar_context_for_prompt()}\n\n"
            f"Уже определено намерение: {intent}.\n"
            f"Логика роутера: {decision.reasoning}\n\n"
            f"{extractor_prompt}"
            f"{_render_intent_context_block(intent_context)}"
        )
        raw_payload = await self._call_llm_json(system, text, step=f"extractor:{intent}")
        raw_payload = _normalize_action_payload(raw_payload)
        # Убедимся, что action_type внутри extractor-payload не съехал.
        if intent == "add_checklist_items":
            raw_payload["action_type"] = "create_checklist_item"
        else:
            raw_payload["action_type"] = intent
        # Перенесём reasoning из роутера, чтобы было видно в логах оркестратора.
        raw_payload.setdefault("reasoning", decision.reasoning)

        try:
            return AgentAction.model_validate(raw_payload)
        except ValidationError as exc:
            logger.warning(
                "intent_extractor validation failed: intent=%s err=%s payload=%s",
                intent,
                exc,
                raw_payload,
            )
            raise OpenRouterLLMError(
                "Модель вернула поля в неожиданном формате. Попробуйте короче или явно укажите карточку.",
            ) from exc

    # ---------------------------------------------------------------- LLM call
    async def _call_llm_json(self, system: str, user: str, *, step: str) -> dict[str, Any]:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            raise OpenRouterLLMError(
                "Модель (OpenRouter/OpenAI): неверный или пустой ключ. Проверьте OPENROUTER_API_KEY или OPENAI_API_KEY.",
            ) from exc
        except RateLimitError as exc:
            raise OpenRouterLLMError(
                "Модель: слишком много запросов. Подождите и повторите.",
            ) from exc
        except APIConnectionError as exc:
            raise OpenRouterLLMError(
                "Модель: нет соединения с OpenRouter/OpenAI (сеть, VPN, OPENROUTER_BASE_URL).",
            ) from exc
        except APIStatusError as exc:
            raise OpenRouterLLMError(
                f"Модель: ошибка API ({exc.status_code}). Проверьте CHAT_MODEL и ключ.",
            ) from exc

        raw = completion.choices[0].message.content or "{}"
        logger.debug("LLM raw response (%s): %s", step, raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenRouterLLMError(
                "Модель вернула не JSON. Уточните запрос или смените CHAT_MODEL.",
            ) from exc
        if not isinstance(payload, dict):
            raise OpenRouterLLMError("Модель вернула не объект JSON.")
        return payload
