"""Тесты по датасету NeuroNudge: жизненные формулировки → целевой intent роутера.

Дальше по этим же сценариям можно:
- расширять CREATE_CARD_MARKERS / промпт роутера;
- поднимать RULE_BASELINE_CREATE_CARD_MATCHES, когда rule-based начинает ловить больше фраз.

LLM-роутер здесь не вызывается — только структура данных и детерминированные правила.
"""

from __future__ import annotations

import pytest

from app.services.intent_rules import quick_classify_intent
from tests.neuronudge_dataset import NEURONUDGE_INTENT_OVERRIDES, load_neuronudge_scenarios, scenario_id


def test_fixture_has_expected_shape():
    scenarios = load_neuronudge_scenarios()
    assert len(scenarios) == 100
    first = scenarios[0]
    assert first.category == "Карьера и Бизнес"
    assert "клиенту" in first.task_title.lower()
    assert len(first.user_text) > 20


def test_intent_override_row_indices_valid():
    """NEURONUDGE_INTENT_OVERRIDES ссылается только на существующие строки датасета."""
    scenarios = load_neuronudge_scenarios()
    n = len(scenarios)
    for idx in NEURONUDGE_INTENT_OVERRIDES:
        assert 0 <= idx < n, f"override index {idx} out of range 0..{n - 1}"


@pytest.mark.parametrize(
    "scenario",
    load_neuronudge_scenarios(),
    ids=scenario_id,
)
def test_user_utterance_is_non_empty_realistic(scenario):
    assert len(scenario.user_text.strip()) >= 12
    assert len(scenario.task_title.strip()) >= 3


def test_neuronudge_no_complete_card_marker_collision():
    """В датасете не должно быть фраз закрытия карточки (иначе rule «complete» перебьёт create)."""
    markers_substrings = (
        "можно закрывать",
        "можно закрыть",
        "закрой карточк",
        "обслужили",
        "уже сделали",
        "перенеси в готово",
    )
    bad: list[tuple[int, str]] = []
    for s in load_neuronudge_scenarios():
        low = s.user_text.lower()
        for m in markers_substrings:
            if m in low:
                bad.append((s.row_index, m))
    assert not bad, f"Неожиданные маркеры закрытия: {bad[:10]}"


def test_quick_classify_create_card_baseline():
    """Сколько разговорных фраз ловит rule-based без LLM (нижняя граница — поднимать при улучшении маркеров)."""
    scenarios = load_neuronudge_scenarios()
    matches = [s for s in scenarios if quick_classify_intent(s.user_text) == "create_card"]
    RULE_BASELINE_CREATE_CARD_MATCHES = 5
    assert len(matches) >= RULE_BASELINE_CREATE_CARD_MATCHES, (
        f"Ожидалось хотя бы {RULE_BASELINE_CREATE_CARD_MATCHES} совпадений create_card по правилам, "
        f"сейчас {len(matches)}. Если снизилось — проверьте регрессию в intent_rules."
    )
