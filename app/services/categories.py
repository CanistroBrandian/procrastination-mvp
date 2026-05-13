from __future__ import annotations

from dataclasses import dataclass

from app.db.models import UserProfile
from app.integrations.trello import TrelloClient


@dataclass(frozen=True)
class CategorySpec:
    key: str
    label_name: str
    color: str
    aliases: tuple[str, ...]


CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec("home", "Home", "green", ("home", "дом", "домаш", "быт")),
    CategorySpec("work", "Work", "blue", ("work", "работ", "job", "офис")),
    CategorySpec("interests", "Interests", "purple", ("interests", "интерес", "хобби")),
    CategorySpec("routine", "Routine", "orange", ("routine", "рутин", "ежеднев", "everyday")),
    CategorySpec("study", "Study", "yellow", ("study", "учеб", "обуч", "английск")),
    CategorySpec("health", "Health", "red", ("health", "здоров", "спорт")),
)

_BY_KEY = {x.key: x for x in CATEGORY_SPECS}


def list_categories_for_user() -> str:
    rows = []
    for spec in CATEGORY_SPECS:
        aliases = ", ".join(spec.aliases[:3])
        rows.append(f"- {spec.key}: {aliases}")
    return "Доступные категории:\n" + "\n".join(rows)


def normalize_category_key(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip().lower().replace("ё", "е")
    if s in _BY_KEY:
        return s
    for spec in CATEGORY_SPECS:
        if any(alias in s for alias in spec.aliases):
            return spec.key
    return None


def detect_category_from_text(text: str | None) -> str | None:
    return normalize_category_key(text)


class CategoryService:
    def __init__(self, trello_client: TrelloClient):
        self.trello_client = trello_client

    async def ensure_label_id(self, board_id: str, category_key: str) -> str:
        key = normalize_category_key(category_key)
        if not key:
            raise ValueError("unknown category")
        spec = _BY_KEY[key]
        labels = await self.trello_client.list_board_labels(board_id)
        for row in labels:
            if str(row.get("name") or "").strip().lower() == spec.label_name.lower():
                return str(row.get("id") or "")
        created = await self.trello_client.create_board_label(board_id, spec.label_name, spec.color)
        return str(created.get("id") or "")

    async def apply_category(self, profile: UserProfile, card_id: str, category_key: str) -> None:
        if not profile.trello_board_id:
            return
        key = normalize_category_key(category_key)
        if not key:
            return
        labels = await self.trello_client.list_board_labels(profile.trello_board_id)
        by_name = {str(x.get("name") or "").strip().lower(): str(x.get("id") or "") for x in labels if x.get("id")}
        category_label_ids = {
            by_name.get(spec.label_name.lower())
            for spec in CATEGORY_SPECS
            if by_name.get(spec.label_name.lower())
        }
        category_label_ids.discard(None)
        card = await self.trello_client.get_card(card_id)
        existing_ids = {str(x) for x in list(card.get("idLabels") or [])}
        target_id = by_name.get(_BY_KEY[key].label_name.lower())
        if not target_id:
            target_id = await self.ensure_label_id(profile.trello_board_id, key)
        for label_id in existing_ids:
            if label_id in category_label_ids and label_id != target_id:
                await self.trello_client.remove_label_from_card(card_id, label_id)
        if target_id and target_id not in existing_ids:
            await self.trello_client.add_label_to_card(card_id, target_id)
