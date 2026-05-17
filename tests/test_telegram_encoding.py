from app.integrations.telegram import _normalize_mojibake


def test_normalize_mojibake_recovers_cyrillic():
    assert _normalize_mojibake("РџСЂРёРЅСЏС‚Рѕ.") == "Принято."


def test_normalize_mojibake_keeps_normal_text():
    text = "Принято, работаем."
    assert _normalize_mojibake(text) == text
