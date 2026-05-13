"""Схемы для двухшаговой агентной системы (router → extractor)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Intent = Literal[
    "create_card",
    "update_card",
    "delete_card",
    "move_card",
    "complete_card_by_text",
    "add_checklist_items",
    "complete_task_from_text",
    "update_checklist_item",
    "add_comment",
    "add_card_label",
    "remove_card_label",
    "add_card_member",
    "remove_card_member",
    "attach_file",
    "set_category",
    "create_routine_template",
    "pause_routine_template",
    "resume_routine_template",
    "list_routine_templates",
    "set_persona",
    "link_board",
    "chitchat",
]


VALID_INTENTS = (
    "create_card",
    "update_card",
    "delete_card",
    "move_card",
    "complete_card_by_text",
    "add_checklist_items",
    "complete_task_from_text",
    "update_checklist_item",
    "add_comment",
    "add_card_label",
    "remove_card_label",
    "add_card_member",
    "remove_card_member",
    "attach_file",
    "set_category",
    "create_routine_template",
    "pause_routine_template",
    "resume_routine_template",
    "list_routine_templates",
    "set_persona",
    "link_board",
    "chitchat",
)


class IntentDecision(BaseModel):
    """Результат шага-роутера: только что хочет пользователь и почему."""

    model_config = ConfigDict(extra="ignore")

    intent: Intent
    reasoning: str = Field(default="")
    response_text: str = Field(default="Принято, работаем.")
