from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.services.categories import CategoryService, normalize_category_key


@dataclass
class FakeProfile:
    trello_board_id: str | None = "B1"


class FakeTrello:
    def __init__(self) -> None:
        self.removed: list[str] = []
        self.added: list[str] = []
        self.created: list[tuple[str, str, str]] = []
        self._labels = [
            {"id": "l-home", "name": "Home", "color": "green"},
            {"id": "l-work", "name": "Work", "color": "blue"},
        ]

    async def list_board_labels(self, board_id: str):  # noqa: ARG002
        return list(self._labels)

    async def create_board_label(self, board_id: str, name: str, color: str = "blue"):  # noqa: ARG002
        new_id = f"l-{name.lower()}"
        self.created.append((board_id, name, color))
        self._labels.append({"id": new_id, "name": name, "color": color})
        return {"id": new_id, "name": name, "color": color}

    async def get_card(self, card_id: str):  # noqa: ARG002
        return {"id": card_id, "idLabels": ["l-home"]}

    async def remove_label_from_card(self, card_id: str, label_id: str):  # noqa: ARG002
        self.removed.append(label_id)
        return {}

    async def add_label_to_card(self, card_id: str, label_id: str):  # noqa: ARG002
        self.added.append(label_id)
        return {}


def _run(coro):
    return asyncio.run(coro)


def test_normalize_category_key_aliases():
    assert normalize_category_key("домашние дела") == "home"
    assert normalize_category_key("work") == "work"
    assert normalize_category_key("английский") == "study"
    assert normalize_category_key("unknown-category") is None


def test_apply_category_replaces_previous_category_label():
    trello = FakeTrello()
    service = CategoryService(trello)
    profile = FakeProfile()

    _run(service.apply_category(profile, "card-1", "work"))

    assert trello.removed == ["l-home"]
    assert trello.added == ["l-work"]

