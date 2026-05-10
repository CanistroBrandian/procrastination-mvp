"""Юнит-тесты rule-based pre-router'а.

Это первый рубеж классификации: дешёвый, детерминированный, без LLM.
Покрывает очевидные случаи на разговорной русской речи (голосовые от
пользователя). Если правило сработало — LLM-роутер не вызывается вообще.
"""

from __future__ import annotations

import pytest

from app.services.intent_rules import quick_classify_intent


# --------------------------------------------------------------------- create


@pytest.mark.parametrize(
    "text",
    [
        "Слушай, давай на завтра поставим задачу делать уроки",
        "Поставь задачу купить молоко",
        "Поставлю задачу заехать в магазин",
        "Создай карточку «обслуживание машины»",
        "Создадим задачу посадить дерево",
        "Запланируй визит к врачу",
        "Не забыть купить билеты",
        "Не забудь позвонить маме",
        "Напомни купить хлеб",
        "Зададим задачу прочитать книгу",
        "Добавь задачу принести зонт",
        "Запиши задачу: убраться в гараже",
    ],
)
def test_rule_classify_create_card(text: str) -> None:
    assert quick_classify_intent(text) == "create_card"


# --------------------------------------------------------------------- complete card


@pytest.mark.parametrize(
    "text",
    [
        "так вы машину мы обслужили в принципе можно закрывать эту задачу",
        "Машину обслужили, можно закрывать",
        "Закрой карточку обслуживание машины",
        "Закрой задачу про машину",
        "Переведи в завершённые карточку машина",
        "Перенеси в готово машину",
        "Карточка готова, переводи",
        "Задача готова",
        "Уже сделали всё по машине",
        "Мы выполнили карточку про машину",
    ],
)
def test_rule_classify_complete_card(text: str) -> None:
    assert quick_classify_intent(text) == "complete_card_by_text"


def test_rule_complete_card_priority_over_create_when_both_present() -> None:
    """Слово «задача» в фразе про закрытие не должно перевесить closing-маркеры."""
    text = "так вы машину мы обслужили в принципе можно закрывать эту задачу"
    assert quick_classify_intent(text) == "complete_card_by_text"


def test_rule_negation_blocks_complete_card() -> None:
    """«ещё не сделали» — НЕ закрытие, должны вернуть None и пустить в LLM."""
    assert quick_classify_intent("мы ещё не сделали машину, не закрывай") is None
    assert quick_classify_intent("пока не обслужили машину") is None


# --------------------------------------------------------------------- delete / move


def test_rule_classify_delete_card() -> None:
    assert quick_classify_intent("удали карточку про машину") == "delete_card"
    assert quick_classify_intent("сотри задачу") == "delete_card"


def test_rule_classify_move_card() -> None:
    assert quick_classify_intent("перенеси в работу карточку про машину") == "move_card"
    assert quick_classify_intent("в работу задачу про отчёт") == "move_card"


# --------------------------------------------------------------------- link / persona


def test_rule_classify_link_board() -> None:
    assert quick_classify_intent(
        "вот моя доска https://trello.com/b/abc123/my-board"
    ) == "link_board"


def test_rule_classify_set_persona() -> None:
    assert quick_classify_intent("будь как мама") == "set_persona"
    assert quick_classify_intent("переключись на дзен") == "set_persona"


# --------------------------------------------------------------------- fallback


@pytest.mark.parametrize(
    "text",
    [
        "Помыл лобовое стекло",
        "Привет, как дела?",
        "Закончил презентацию для проекта А",
        "Что у меня по делам на сегодня?",
        "",
        "   ",
    ],
)
def test_rule_returns_none_for_ambiguous_or_subtask(text: str) -> None:
    """Случаи, где правил недостаточно — нужно идти в LLM."""
    assert quick_classify_intent(text) is None
