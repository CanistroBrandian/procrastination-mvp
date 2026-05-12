from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.services.intent_rules import quick_classify_intent


class FlowDecision(str, Enum):
    CONTINUE_CURRENT_FLOW = "continue_current_flow"
    SWITCH_FLOW_WITH_CONFIRM = "switch_flow_with_confirm"
    FRESH_FLOW = "fresh_flow"


class PendingFlow(str, Enum):
    CREATE_CARD = "create_card"
    CREATE_CHECKLIST_ITEM = "create_checklist_item"
    ADD_CHECKLIST_PICK_CARD = "add_checklist_pick_card"
    UPDATE_CARD_PICK_CARD = "update_card_pick_card"
    COMPLETE_TASK = "complete_task"
    COMPLETE_CARD = "complete_card"
    COMPLETE_CARD_FORCE = "complete_card_force"
    CONFIRM_FLOW_SWITCH = "confirm_flow_switch"


@dataclass(frozen=True)
class DecisionInput:
    user_text: str
    expected_intents: set[str]
    current_flow: PendingFlow | None = None


def _looks_like_new_command(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s or len(s) <= 3:
        return False
    markers = (
        "постав",
        "создай",
        "запланир",
        "напомни",
        "не забы",
        "закрой",
        "удали",
        "сотри",
        "перенес",
        "обнови",
        "переименуй",
        "новая задач",
        "новую задач",
        "новое задание",
        "сделай задач",
    )
    return any(m in s for m in markers)


def decide_flow_action(payload: DecisionInput) -> FlowDecision:
    ruled = quick_classify_intent(payload.user_text)
    if ruled is not None and ruled not in payload.expected_intents:
        return FlowDecision.SWITCH_FLOW_WITH_CONFIRM
    if _looks_like_new_command(payload.user_text):
        return FlowDecision.SWITCH_FLOW_WITH_CONFIRM
    return FlowDecision.CONTINUE_CURRENT_FLOW
