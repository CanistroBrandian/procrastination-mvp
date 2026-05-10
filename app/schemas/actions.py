from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ActionType = Literal[
    "create_card",
    "update_card",
    "delete_card",
    "move_card",
    "create_checklist_item",
    "update_checklist_item",
    "complete_task_from_text",
    "complete_card_by_text",
    "add_comment",
    "add_card_label",
    "remove_card_label",
    "add_card_member",
    "remove_card_member",
    "attach_file",
    "ask_for_clarification",
    "set_persona",
    "link_board",
    "none",
]


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reasoning: str | None = None
    action_type: ActionType
    card_id: str | None = None
    list_name: str | None = None
    card_name: str | None = None
    card_description: str | None = None
    # ISO 8601 для Trello: due / start; снятие срока — отдельным сценарием (не в MVP).
    due: str | None = None
    start: str | None = None
    due_complete: bool | None = None
    closed: bool | None = None
    # Создание: pos в API — top | bottom | число.
    position: str | float | None = None
    checklist_name: str | None = None
    checklist_item: str | None = None
    # Phase 2: id чеклиста и пункта в Trello (не имя колонки доски).
    checklist_id: str | None = None
    check_item_id: str | None = None
    check_item_complete: bool | None = None
    check_item_new_name: str | None = None
    comment_text: str | None = None
    label_id: str | None = None
    member_id: str | None = None
    # Для complete_task_from_text — что пользователь утверждает, что выполнил.
    match_text: str | None = None
    file_url: str | None = None
    question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_payload(cls, data: Any) -> Any:
        """До разбора полей: null metadata, checklist_item-массив из модели."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if out.get("metadata") is None:
            out["metadata"] = {}
        ci = out.get("checklist_item")
        if isinstance(ci, list):
            parts = [str(x).strip() for x in ci if x is not None and str(x).strip()]
            out["checklist_item"] = "; ".join(parts) if parts else None
        return out

    @staticmethod
    def _normalize_textish(v: Any) -> str | None:
        """LLM иногда отдаёт список пунктов вместо одной строки."""
        if v is None:
            return None
        if isinstance(v, list):
            parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
            return "; ".join(parts) if parts else None
        if isinstance(v, dict):
            return None
        s = str(v).strip()
        return s or None

    @field_validator(
        "card_name",
        "card_description",
        "checklist_name",
        "checklist_item",
        "check_item_new_name",
        "comment_text",
        "match_text",
        "question",
        mode="before",
    )
    @classmethod
    def coerce_string_fields(cls, v: Any) -> str | None:
        return cls._normalize_textish(v)

    @field_validator("due_complete", "closed", "check_item_complete", mode="before")
    @classmethod
    def coerce_optional_bool(cls, v: Any) -> bool | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            low = v.strip().lower()
            if low in ("true", "1", "yes", "да"):
                return True
            if low in ("false", "0", "no", "нет"):
                return False
        return None

    @field_validator("position", mode="before")
    @classmethod
    def coerce_position(cls, v: Any) -> str | float | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return float(v)
        if isinstance(v, float):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("top", "bottom"):
                return s
            try:
                return float(s)
            except ValueError:
                return None
        return None

    @field_validator("metadata", mode="before")
    @classmethod
    def coerce_metadata(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {}


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: AgentAction
    response_text: str = "Принято, работаем."

    @field_validator("response_text", mode="before")
    @classmethod
    def response_text_non_null(cls, v: Any) -> str:
        if v is None:
            return "Принято, работаем."
        s = str(v).strip()
        return s or "Принято, работаем."
