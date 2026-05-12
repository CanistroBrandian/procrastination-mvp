from __future__ import annotations

import re
from dataclasses import dataclass

from app.db.models import UserProfile
from app.integrations.trello import TrelloClient
from app.schemas.actions import AgentAction
from app.services.agent import AgentService
from app.services.card_search import CardSearchHit, confident_unique_hit, format_cards_for_search, merge_search_results
from app.services.checklist_match import CardRef, best_card_matches, cards_from_payload
from app.services.trello_board_filters import drop_cards_in_archived_lists


@dataclass(frozen=True)
class CardResolution:
    action: AgentAction
    resolved_card_id: str | None = None
    resolved_card_name: str | None = None
    confidence: float | None = None
    question: str | None = None


def _infer_card_title_from_text(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    patterns = (
        r'«([^»]{2,160})»',
        r'"([^"]{2,160})"',
    )
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1).strip()
    return None


def _extract_rename_old_name(text: str) -> str | None:
    s = text or ""
    m = re.search(r"переимен\w*.*?«([^»]{2,160})».*?\b(?:на|в)\b", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'переимен\w*.*?"([^"]{2,160})".*?\b(?:на|в)\b', s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


class CardResolver:
    def __init__(self, trello_client: TrelloClient, agent_service: AgentService):
        self.trello_client = trello_client
        self.agent_service = agent_service

    async def _search_cards_hybrid(
        self,
        profile: UserProfile,
        query: str,
        cards: list[CardRef],
    ) -> list[CardSearchHit]:
        if not cards or not query:
            return []
        fuzzy = best_card_matches(query, cards)
        if fuzzy:
            top_ref, top_score = fuzzy[0]
            second_score = fuzzy[1][1] if len(fuzzy) > 1 else 0.0
            if top_score >= 0.7 and (top_score - second_score) >= 0.18:
                return [
                    CardSearchHit(
                        card_id=top_ref.card_id,
                        card_name=top_ref.card_name,
                        score=min(1.0, top_score),
                        reason="по названию",
                        incomplete_items=top_ref.incomplete_items,
                        total_items=top_ref.total_items,
                        list_id=top_ref.list_id,
                    ),
                ]
        cards_payload = format_cards_for_search(cards)
        try:
            semantic = await self.agent_service.search_cards_semantic(query, cards_payload, persona=profile.persona.value)
        except Exception:  # noqa: BLE001
            semantic = []
        return merge_search_results(fuzzy, semantic, cards)

    async def _load_board_cards(self, profile: UserProfile, *, exclude_done: bool) -> list[CardRef]:
        if not profile.trello_board_id:
            return []
        lists_raw = await self.trello_client.list_lists(profile.trello_board_id)
        payload = await self.trello_client.list_board_cards_with_checklists(profile.trello_board_id)
        payload = drop_cards_in_archived_lists(payload, lists_raw)
        cards = cards_from_payload(payload)
        if exclude_done and profile.trello_done_list_id:
            cards = [c for c in cards if c.list_id != profile.trello_done_list_id]
        return cards

    async def resolve_for_action(
        self,
        *,
        profile: UserProfile,
        user_text: str,
        action: AgentAction,
        active_card_id: str | None = None,
        active_card_name: str | None = None,
        exclude_done: bool = False,
    ) -> CardResolution:
        if action.card_id:
            return CardResolution(
                action=action,
                resolved_card_id=action.card_id,
                resolved_card_name=action.card_name,
                confidence=1.0,
            )
        cards = await self._load_board_cards(profile, exclude_done=exclude_done)
        if not cards:
            return CardResolution(action=action, question="Не вижу доступных карточек на доске. Уточните название карточки.")

        explicit_name = _extract_rename_old_name(user_text) or _infer_card_title_from_text(user_text)
        query = explicit_name or action.card_name or active_card_name or (action.metadata or {}).get("source_card_name")
        query = str(query or "").strip()
        if not query and active_card_id:
            active = next((c for c in cards if c.card_id == active_card_id), None)
            if active is not None:
                return CardResolution(
                    action=action.model_copy(update={"card_id": active.card_id}),
                    resolved_card_id=active.card_id,
                    resolved_card_name=active.card_name,
                    confidence=0.9,
                )
        if not query:
            return CardResolution(action=action, question="Уточните, какую карточку нужно изменить: напишите ее название.")

        hits = await self._search_cards_hybrid(profile, query, cards)
        winner = confident_unique_hit(hits)
        if winner is not None:
            return CardResolution(
                action=action.model_copy(update={"card_id": winner.card_id}),
                resolved_card_id=winner.card_id,
                resolved_card_name=winner.card_name,
                confidence=winner.score,
            )
        if not hits:
            return CardResolution(action=action, question=f"Не нашел карточку по запросу «{query}». Уточните название.")

        cands = [{"card_id": h.card_id, "card_name": h.card_name} for h in hits[:5]]
        listing = "\n".join(f"{i + 1}) {c['card_name']}" for i, c in enumerate(cands))
        ask = (
            f"Нашел несколько карточек для «{query}». Выберите номер:\n{listing}"
        )
        return CardResolution(
            action=action.model_copy(
                update={
                    "action_type": "ask_for_clarification",
                    "question": ask,
                    "metadata": {
                        **(action.metadata or {}),
                        "after_clarification": "pick_card_for_action",
                        "candidates": cands,
                        "resume_action": action.model_dump(mode="json"),
                    },
                },
            ),
            question=ask,
        )
