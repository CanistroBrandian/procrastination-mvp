"""Загрузка датасета жизненных сценариев NeuroNudge (Excel → JSON).

Исходник: NeuroNudge_Intents_Dataset.xlsx (колонки: категория, сфера, пример задачи,
разговорная реплика пользователя). Фикстура: tests/fixtures/neuronudge_intents_dataset.json

Контракт для тестов роутера: целевой intent для «зафиксировать задачу» — create_card.
Исключения задаются в NEURONUDGE_INTENT_OVERRIDES по индексу строки (0-based по данным без шапки).

Перегенерация JSON: установите openpyxl и экспортируйте первый лист Excel в UTF-8 JSON
в tests/fixtures/neuronudge_intents_dataset.json (как при первом импорте датасета).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.schemas.intents import Intent

FIXTURE_PATH: Final = Path(__file__).resolve().parent / "fixtures" / "neuronudge_intents_dataset.json"

# Если для сценария целевой intent не create_card — указать здесь (индекс строки данных, без заголовка Excel).
NEURONUDGE_INTENT_OVERRIDES: dict[int, Intent] = {}


@dataclass(frozen=True)
class NeuronudgeScenario:
    """Одна строка датасета."""

    row_index: int
    category: str
    sphere: str
    task_title: str
    user_text: str

    @property
    def expected_router_intent(self) -> Intent:
        return NEURONUDGE_INTENT_OVERRIDES.get(self.row_index, "create_card")


def load_neuronudge_scenarios(*, fixture_path: Path | None = None) -> list[NeuronudgeScenario]:
    path = fixture_path or FIXTURE_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return []
    header, *rows = raw
    if len(header) != 4:
        raise ValueError("Ожидалась строка заголовка из 4 колонок")
    out: list[NeuronudgeScenario] = []
    for i, row in enumerate(rows):
        if row is None or len(row) != 4:
            raise ValueError(f"Строка {i}: ожидалось 4 колонки, получено {row!r}")
        cat, sphere, title, text = row
        out.append(
            NeuronudgeScenario(
                row_index=i,
                category=str(cat or "").strip(),
                sphere=str(sphere or "").strip(),
                task_title=str(title or "").strip(),
                user_text=str(text or "").strip(),
            )
        )
    return out


def scenario_id(scenario: NeuronudgeScenario, *, max_slug: int = 48) -> str:
    """Короткий id для pytest parametrize."""
    slug = re.sub(r"\s+", "_", scenario.task_title.lower())
    slug = re.sub(r"[^\w\u0400-\u04FF_-]+", "", slug, flags=re.UNICODE)
    slug = slug.strip("_")[:max_slug] or "scenario"
    return f"{scenario.row_index:03d}_{slug}"
