from __future__ import annotations

from app.services import agent as agent_module


def test_agent_fallback_response_text_is_readable_russian():
    payload = agent_module._normalize_action_payload({})  # noqa: SLF001
    assert payload["response_text"] == "Принято, работаем."

    router_payload = agent_module._normalize_router_payload({})  # noqa: SLF001
    assert router_payload["response_text"] == "Принято, работаем."


def test_router_prompt_contains_new_intents():
    text = agent_module.ROUTER_PROMPT
    for marker in (
        "set_category",
        "create_routine_template",
        "pause_routine_template",
        "resume_routine_template",
        "list_routine_templates",
    ):
        assert marker in text

