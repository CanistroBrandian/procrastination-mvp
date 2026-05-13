"""Rule-based pre-router интентов.

Зачем нужен: LLM-роутер периодически ошибается на разговорных голосовых
репликах. Например, «машину обслужили, можно закрывать эту задачу» слабая
модель может неверно посчитать за `create_card` (видя слово «задача»).
Для очевидных случаев нет смысла платить latency и риск ошибки LLM —
делаем быструю детерминированную классификацию по ключевым маркерам.

Контракт:
- `quick_classify_intent(text)` возвращает либо известный intent, либо None.
- Если вернуло None — пусть LLM-роутер решает.
- Приоритеты заточены под продуктовый сценарий: ЗАКРЫТИЕ карточки имеет
  приоритет над созданием (фраза «обслужили, можно закрывать ЗАДАЧУ» — это
  закрытие, хотя слово «задача» тоже встречается).

Не покрывает:
- complete_task_from_text (отметить ОДИН пункт чеклиста) — нет надёжного
  поверхностного маркера, нужен контекст времени глагола; оставляем LLM.
- update_card / move_card / add_comment / metadata-операции — слишком
  богатый контекст; LLM здесь точнее.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from app.schemas.intents import Intent


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- маркеры
# Все строки сравниваются в нижнем регистре, как СУБСТРОКИ — это устойчиво
# к падежам и формам слов («обслужили», «обслуживанием», «закрывайте»).

LINK_BOARD_MARKERS: tuple[str, ...] = (
    "trello.com/b/",
)

PERSONA_MARKERS: tuple[str, ...] = (
    "будь как мам", "переключись на мам", "режим мам",
    "будь как илон", "переключись на илон", "режим илон",
    "будь как дзен", "переключись на дзен", "режим zen",
)

# Закрытие/завершение ВСЕЙ карточки. Идёт первым, чтобы перебивать «задача».
COMPLETE_CARD_MARKERS: tuple[str, ...] = (
    "можно закрывать",
    "можно закрыть",
    "закрой карточк",
    "закрой задач",
    "закрой эту задач",
    "закрой это задани",
    "закрывай эту задач",
    "закрывай задач",
    "закрывайте задач",
    "перевести в завершен",
    "перевести в завершён",
    "переведи в завершен",
    "переведи в завершён",
    "перенеси в завершен",
    "перенеси в завершён",
    "перенеси в готово",
    "перенеси в done",
    "переведи в done",
    "карточка готова",
    "задача готова",
    "задача завершен",
    "карточка завершен",
    "обслужили",
    "уже сделали",
    "мы сделали",
    "уже выполнили",
    "мы выполнили",
    "уже завершили",
    "мы завершили",
)

# Удаление целой карточки.
DELETE_CARD_MARKERS: tuple[str, ...] = (
    "удали карточк",
    "удали задач",
    "сотри карточк",
    "сотри задач",
    "убери карточк",
    "убери задач",
)

# Создание новой задачи. Слова из ИМПЕРАТИВА/БУДУЩЕГО (не «сделал»).
CREATE_CARD_MARKERS: tuple[str, ...] = (
    "поставь задач",
    "поставим задач",
    "поставлю задач",
    "поставить задач",
    "ставь задач",
    "ставим задач",
    "создай задач",
    "создай карточк",
    "создадим задач",
    "сделай задач",
    "запиши задач",
    "запиши в задач",
    "запланируй",
    "не забыть",
    "не забудь",
    "напомни мне",
    "напомни купить",
    "напомни сделать",
    "новая задач",
    "новую задач",
    "новое задание",
    "добавь задач",
    "зададим задач",
    "задай задач",
    "задам задач",
)

# Перенос карточки в другую колонку (не закрытие).
MOVE_CARD_MARKERS: tuple[str, ...] = (
    "перенеси в работу",
    "перенеси в doing",
    "перенеси в инбокс",
    "перенеси в inbox",
    "переведи в работу",
    "переведи в doing",
    "в работу карточк",
    "в работу задач",
)

SET_CATEGORY_MARKERS: tuple[str, ...] = (
    "категори",
    "category",
    "пометь как",
)

CREATE_ROUTINE_MARKERS: tuple[str, ...] = (
    "рутин",
    "ежедневн",
    "каждый день",
    "шаблон рутины",
)

PAUSE_ROUTINE_MARKERS: tuple[str, ...] = (
    "пауза рутин",
    "останови рутин",
    "выключи рутин",
)

RESUME_ROUTINE_MARKERS: tuple[str, ...] = (
    "возобнови рутин",
    "включи рутин",
    "продолжи рутин",
)

LIST_ROUTINE_MARKERS: tuple[str, ...] = (
    "список рутин",
    "покажи рутины",
    "какие рутины",
)


# ---------------------------------------------------------------- helpers


def _has_any(text: str, markers: Iterable[str]) -> bool:
    return any(m in text for m in markers)


# Регекс для отрицания «не закрыли», «ещё не сделали» и т.п. — НЕ закрытие.
_NEGATION_BEFORE_DONE = re.compile(
    r"(не\s+сделал|не\s+обслужил|ещё\s+не|еще\s+не|пока\s+не)\b",
    flags=re.IGNORECASE,
)


def quick_classify_intent(text: str) -> Intent | None:
    """Быстрая детерминированная классификация. None — пусть решает LLM."""
    s = (text or "").strip().lower()
    if not s:
        return None

    # Привязка доски — однозначный маркер.
    if _has_any(s, LINK_BOARD_MARKERS):
        return "link_board"

    if _has_any(s, PERSONA_MARKERS):
        return "set_persona"

    if _has_any(s, LIST_ROUTINE_MARKERS):
        return "list_routine_templates"
    if _has_any(s, PAUSE_ROUTINE_MARKERS):
        return "pause_routine_template"
    if _has_any(s, RESUME_ROUTINE_MARKERS):
        return "resume_routine_template"
    if _has_any(s, CREATE_ROUTINE_MARKERS):
        return "create_routine_template"
    if _has_any(s, SET_CATEGORY_MARKERS):
        return "set_category"

    # ВАЖНО: закрытие проверяем РАНЬШЕ создания. Фраза «можно закрывать ЭТУ
    # ЗАДАЧУ» содержит «задача», но это не create.
    if _has_any(s, COMPLETE_CARD_MARKERS):
        # Защита от отрицания: «ещё не сделали» → НЕ закрытие.
        if _NEGATION_BEFORE_DONE.search(s):
            logger.info("rule-based: closing markers seen but negation present, fallback to LLM")
            return None
        return "complete_card_by_text"

    if _has_any(s, DELETE_CARD_MARKERS):
        return "delete_card"

    if _has_any(s, MOVE_CARD_MARKERS):
        return "move_card"

    if _has_any(s, CREATE_CARD_MARKERS):
        return "create_card"

    return None
