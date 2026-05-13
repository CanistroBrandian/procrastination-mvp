"""Р”РІСѓС…С€Р°РіРѕРІС‹Р№ LLM-Р°РіРµРЅС‚: РєР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ РЅР°РјРµСЂРµРЅРёР№ + СѓР·РєРёР№ РёР·РІР»РµРєР°С‚РµР»СЊ РїРѕР»РµР№.

РђСЂС…РёС‚РµРєС‚СѓСЂР°:
  1) IntentRouter вЂ” РєСЂРѕС€РµС‡РЅС‹Р№ РїСЂРѕРјРїС‚. РўРѕР»СЊРєРѕ РІС‹Р±РёСЂР°РµС‚ РѕРґРёРЅ РёР· ~17 РёРЅС‚РµРЅС‚РѕРІ.
     Р§С‘С‚РєРёРµ РјР°СЂРєРµСЂС‹ РїРѕ РІСЂРµРјРµРЅРё РіР»Р°РіРѕР»РѕРІ Рё СЂРѕР»Рё (РєР°СЂС‚РѕС‡РєР° vs РїСѓРЅРєС‚ С‡РµРєР»РёСЃС‚Р°).
  2) IntentExtractor вЂ” СѓР·РєРёР№ РїСЂРѕРјРїС‚ РїРѕРґ РІС‹Р±СЂР°РЅРЅС‹Р№ intent. Р—Р°РїРѕР»РЅСЏРµС‚ С‚РѕР»СЊРєРѕ
     СЂРµР»РµРІР°РЅС‚РЅС‹Рµ РїРѕР»СЏ. РќРµ Р·РЅР°РµС‚ РїСЂРѕ РґСЂСѓРіРёРµ РёРЅС‚РµРЅС‚С‹, РїРѕСЌС‚РѕРјСѓ РЅРµ РїСѓС‚Р°РµС‚СЃСЏ.
  3) AgentService.infer_action РІРѕР·РІСЂР°С‰Р°РµС‚ РїСЂРµР¶РЅРёР№ AgentResult вЂ” РѕСЂРєРµСЃС‚СЂР°С‚РѕСЂ
     РЅРµ РЅСѓР¶РґР°РµС‚СЃСЏ РІ РїСЂР°РІРєР°С….

РџСЂРµРёРјСѓС‰РµСЃС‚РІР° РїРѕ СЃСЂР°РІРЅРµРЅРёСЋ СЃРѕ СЃС‚Р°СЂС‹Рј РѕРґРЅРѕС€Р°РіРѕРІС‹Рј РїСЂРѕРјРїС‚РѕРј:
- РљР°Р¶РґС‹Р№ РїСЂРѕРјРїС‚ РєРѕРјРїР°РєС‚РµРЅ Рё РѕРґРЅРѕР·РЅР°С‡РµРЅ в†’ РІС‹С€Рµ С‚РѕС‡РЅРѕСЃС‚СЊ РЅР° СЃР»Р°Р±С‹С… РјРѕРґРµР»СЏС….
- Р›РѕРіРёРєР° РІС‹Р±РѕСЂР° РёРЅС‚РµРЅС‚Р° РѕС‚РґРµР»РµРЅР° РѕС‚ РёР·РІР»РµС‡РµРЅРёСЏ РґР°РЅРЅС‹С…, РїСЂРѕС‰Рµ РѕС‚Р»Р°Р¶РёРІР°С‚СЊ.
- РњРѕР¶РЅРѕ РїРѕРґРјРµРЅСЏС‚СЊ/РґРѕР±Р°РІР»СЏС‚СЊ РёРЅС‚РµРЅС‚С‹, РЅРµ СЂР°Р·СЂР°СЃС‚Р°СЏ РѕРґРёРЅ РіРёРіР°РЅС‚СЃРєРёР№ prompt.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from pydantic import ValidationError

from app.core.errors import OpenRouterLLMError
from app.schemas.actions import AgentAction, AgentResult
from app.schemas.intents import VALID_INTENTS, IntentDecision
from app.services.card_search import CardSearchHit
from app.services.date_infer import calendar_context_for_prompt
from app.services.intent_rules import quick_classify_intent


logger = logging.getLogger(__name__)


PERSONA_PROMPTS = {
    "elon": (
        "РўС‹ СЃС‚СЂРѕРіРёР№ РјРѕС‚РёРІР°С‚РѕСЂ. РџРёС€Рё РєРѕСЂРѕС‚РєРѕ, СЌРЅРµСЂРіРёС‡РЅРѕ, Р±РµР· РѕСЃРєРѕСЂР±Р»РµРЅРёР№. "
        "Р¤РѕРєСѓСЃРёСЂСѓР№СЃСЏ РЅР° С†РµРЅРµ Р±РµР·РґРµР№СЃС‚РІРёСЏ Рё СЃСЂРѕС‡РЅРѕСЃС‚Рё РїРµСЂРІРѕРіРѕ С€Р°РіР°."
    ),
    "zen": (
        "РўС‹ СЃРїРѕРєРѕР№РЅС‹Р№ РґР·РµРЅ-РЅР°СЃС‚Р°РІРЅРёРє. РџРѕРјРѕРіР°РµС€СЊ РЅР°Р№С‚Рё РїСЂРµРїСЏС‚СЃС‚РІРёРµ Рё РІС‹Р±СЂР°С‚СЊ РјР°Р»РµРЅСЊРєРёР№ СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі."
    ),
    "mom": (
        "РўС‹ РїРѕРґРґРµСЂР¶РёРІР°СЋС‰РёР№ Рё С‚РµРїР»С‹Р№ РїРѕРјРѕС‰РЅРёРє. Р’РµСЂРёС€СЊ РІ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ Рё РґР°РµС€СЊ РєРѕРЅСЃС‚СЂСѓРєС‚РёРІРЅСѓСЋ РїРѕРґРґРµСЂР¶РєСѓ."
    ),
}


# ============================================================================
# 1) ROUTER PROMPT вЂ” С‚РѕР»СЊРєРѕ РєР»Р°СЃСЃРёС„РёРєР°С†РёСЏ intent
# ============================================================================

ROUTER_PROMPT = """\
РўС‹ вЂ” РєР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ РЅР°РјРµСЂРµРЅРёР№ РґР»СЏ Р·Р°РґР°С‡РЅРёРєР° РЅР° Р±Р°Р·Рµ Trello.
РќР° РІС…РѕРґ РїРѕР»СѓС‡Р°РµС€СЊ СЂРµРїР»РёРєСѓ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ (СЂСѓСЃСЃРєРёР№, СЂР°Р·РіРѕРІРѕСЂРЅС‹Р№, РјРѕР¶РµС‚ Р±С‹С‚СЊ С‚СЂР°РЅСЃРєСЂРёР±РёСЂРѕРІР°РЅРѕ РёР· РіРѕР»РѕСЃР°).
РўРІРѕСЏ Р•Р”РРќРЎРўР’Р•РќРќРђРЇ Р·Р°РґР°С‡Р° вЂ” РѕРїСЂРµРґРµР»РёС‚СЊ, РєР°РєРѕРµ РґРµР№СЃС‚РІРёРµ С…РѕС‡РµС‚ СЃРѕРІРµСЂС€РёС‚СЊ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ.

Р’РµСЂРЅРё РЎРўР РћР“Рћ JSON: {"intent":"<РѕРґРёРЅ РёР· СЃРїРёСЃРєР°>","reasoning":"<2-3 С„СЂР°Р·С‹>","response_text":"<РєСЂР°С‚РєРёР№ РѕС‚РІРµС‚ РїРѕР»СЊР·РѕРІР°С‚РµР»СЋ>"}

РРЅС‚РµРЅС‚С‹ (РІС‹Р±РµСЂРё РћР”РРќ):

1) create_card вЂ” РїРѕСЃС‚Р°РІРёС‚СЊ РќРћР’РЈР® Р·Р°РґР°С‡Сѓ.
   Р–РЃРЎРўРљРР™ РџР РРћР РРўР•Рў: РµСЃР»Рё РІ СЂРµРїР»РёРєРµ РµСЃС‚СЊ В«РїРѕСЃС‚Р°РІРёРј Р·Р°РґР°С‡СѓВ», В«РїРѕСЃС‚Р°РІСЊ Р·Р°РґР°С‡СѓВ», В«Р·Р°РґР°РґРёРј Р·Р°РґР°С‡СѓВ»,
   В«Р·Р°РїР»Р°РЅРёСЂСѓР№ Р·Р°РґР°С‡СѓВ», В«РґР°РІР°Р№ Р·Р°РґР°С‡СѓВ», В«РЅРѕРІР°СЏ Р·Р°РґР°С‡Р°В», В«РЅР°РїРѕРјРЅРёВ», В«РЅРµ Р·Р°Р±С‹С‚СЊВ» вЂ” СЌС‚Рѕ Р’РЎР•Р“Р”Рђ create_card,
   Р”РђР–Р• РµСЃР»Рё СЂСЏРґРѕРј РµСЃС‚СЊ РіР»Р°РіРѕР» В«СЃРґРµР»Р°С‚СЊВ» / В«РґРµР»Р°С‚СЊВ» / В«РїРѕРјС‹С‚СЊВ» РІ РёРЅС„РёРЅРёС‚РёРІРµ (СЌС‚Рѕ С‚РѕРіРґР° РЅРµ В«РІС‹РїРѕР»РЅРµРЅРѕВ», Р° РѕРїРёСЃР°РЅРёРµ СЃСѓС‚Рё РЅРѕРІРѕР№ Р·Р°РґР°С‡Рё).
   РњР°СЂРєРµСЂС‹: Р±СѓРґСѓС‰РµРµ РІСЂРµРјСЏ + РЅРѕРІР°СЏ Р°РєС‚РёРІРЅРѕСЃС‚СЊ.
   РџСЂРёРјРµСЂС‹:
     "РЎР»СѓС€Р°Р№, РґР°РІР°Р№ РЅР° Р·Р°РІС‚СЂР° РїРѕСЃС‚Р°РІРёРј Р·Р°РґР°С‡Сѓ РґРµР»Р°С‚СЊ СѓСЂРѕРєРё" в†’ create_card (card_name="РЎРґРµР»Р°С‚СЊ СѓСЂРѕРєРё").
     "Р”Р°РІР°Р№ РЅР°Р·РѕРІРµРј РїРѕСЃС‚Р°РІРёРј Р·Р°РґР°С‡Сѓ, РЅР°РґРѕ РєРѕСЂРѕС‡Рµ СѓСЂРѕРєРё РјРЅРµ СЃРґРµР»Р°С‚СЊ Рё С‚Р°Рј РјР°С‚РµРјР°С‚РёРєР°, СЂСѓСЃСЃРєРёР№ СЏР·С‹Рє Рё РіРµРѕРіСЂР°С„РёСЋ"
        в†’ create_card (card_name="РЎРґРµР»Р°С‚СЊ СѓСЂРѕРєРё", checklist=["РњР°С‚РµРјР°С‚РёРєР°","Р СѓСЃСЃРєРёР№ СЏР·С‹Рє","Р“РµРѕРіСЂР°С„РёСЏ"]).
     "Р—Р°РІС‚СЂР° Р·Р°РґР°РґРёРј Р·Р°РґР°С‡Сѓ РїРѕСЃР°РґРёС‚СЊ РґРµСЂРµРІРѕ, РЅСѓР¶РЅРѕ РєСѓРїРёС‚СЊ РіСЂСѓРЅС‚ Рё Р»РѕРїР°С‚Сѓ" в†’ create_card.
     "РќР°РїРѕРјРЅРё РєСѓРїРёС‚СЊ С…Р»РµР±" в†’ create_card.

2) complete_task_from_text вЂ” РѕС‚РјРµС‚РёС‚СЊ РћР”РРќ РџРЈРќРљРў РІРЅСѓС‚СЂРё СѓР¶Рµ СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµР№ Р·Р°РґР°С‡Рё РєР°Рє Р’Р«РџРћР›РќР•РќРќР«Р™.
   РњР°СЂРєРµСЂС‹: "СЏ СЃРґРµР»Р°Р»", "СЏ РїРѕРјС‹Р»", "Р·Р°РєРѕРЅС‡РёР» РҐ", "РїСЂРѕРїС‹Р»РµСЃРѕСЃРёР»", "РіРѕС‚РѕРІРѕ: Y", "РїРѕРјС‹Р» РѕРєРЅР°".
   РљР РРўРР§РќРћ: РіР»Р°РіРѕР» РІ РџР РћРЁР•Р”РЁР•Рњ РІСЂРµРјРµРЅРё, РѕРїРёСЃС‹РІР°РµС‚ РјРµР»РєСѓСЋ РїРѕРґР·Р°РґР°С‡Сѓ, РЅРµ РІСЃСЋ Р·Р°РґР°С‡Сѓ С†РµР»РёРєРѕРј.
   РџСЂРёРјРµСЂС‹:
     "РџСЂРѕРїС‹Р»РµСЃРѕСЃРёР» РІ РјР°С€РёРЅРµ" в†’ complete_task_from_text.
     "РџРѕРјС‹Р» РѕРєРЅР°" в†’ complete_task_from_text.
     "Р—Р°РєРѕРЅС‡РёР» РїСЂРµР·РµРЅС‚Р°С†РёСЋ РґР»СЏ РїСЂРѕРµРєС‚Р° Рђ" в†’ complete_task_from_text.

3) complete_card_by_text вЂ” Р·Р°РєСЂС‹С‚СЊ/Р·Р°РІРµСЂС€РёС‚СЊ Р’РЎР® РєР°СЂС‚РѕС‡РєСѓ (РѕСЃРЅРѕРІРЅСѓСЋ Р·Р°РґР°С‡Сѓ).
   РњР°СЂРєРµСЂС‹: "Р·Р°РєСЂРѕР№ РєР°СЂС‚РѕС‡РєСѓ РҐ", "РїРµСЂРµРІРµРґРё Р·Р°РґР°С‡Сѓ РҐ РІ Р·Р°РІРµСЂС€С‘РЅРЅС‹Рµ", "РҐ РіРѕС‚РѕРІР°, Р·Р°РєСЂС‹РІР°Р№",
   "РѕР±СЃР»СѓР¶РёРІР°РЅРёРµ РјР°С€РёРЅС‹ вЂ” РіРѕС‚РѕРІРѕ", "РјР°С€РёРЅСѓ РѕР±СЃР»СѓР¶РёР»Рё, РјРѕР¶РЅРѕ Р·Р°РєСЂС‹РІР°С‚СЊ СЌС‚Сѓ Р·Р°РґР°С‡Сѓ".
   РљР РРўРР§РќРћ: РіРѕРІРѕСЂСЏС‚ Рѕ Р¦Р•Р›РћР™ Р·Р°РґР°С‡Рµ, Р° РЅРµ Рѕ РїСѓРЅРєС‚Рµ РІРЅСѓС‚СЂРё РЅРµС‘.
   РџСЂРёРјРµСЂС‹:
     "РњР°С€РёРЅСѓ РјС‹ РѕР±СЃР»СѓР¶РёР»Рё, РєРѕСЂРѕС‡Рµ РјРѕР¶РЅРѕ Р·Р°РєСЂС‹РІР°С‚СЊ СЌС‚Сѓ Р·Р°РґР°С‡Сѓ" в†’ complete_card_by_text.
     "РљР°СЂС‚РѕС‡РєР° РѕР±СЃР»СѓР¶РёРІР°РЅРёРµ РјР°С€РёРЅС‹ вЂ” РїРµСЂРµРІРµРґРё РІ Р·Р°РІРµСЂС€С‘РЅРЅС‹Рµ" в†’ complete_card_by_text.

4) update_card вЂ” РёР·РјРµРЅРёС‚СЊ РїРѕР»Рµ СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµР№ РєР°СЂС‚РѕС‡РєРё (СЃСЂРѕРє, РЅР°Р·РІР°РЅРёРµ, РѕРїРёСЃР°РЅРёРµ, РґР°С‚Р° РЅР°С‡Р°Р»Р°).
   РњР°СЂРєРµСЂС‹: "РїРµСЂРµРЅРµСЃРё РЅР° РїСЏС‚РЅРёС†Сѓ", "РїРµСЂРµРёРјРµРЅСѓР№ РҐ РІ Y", "РїРѕСЃС‚Р°РІСЊ СЃСЂРѕРє 10 РјР°СЏ РЅР° РҐ", "СЃРґРµР»Р°Р№ РҐ РЅР° Р·Р°РІС‚СЂР°".

5) delete_card вЂ” СѓРґР°Р»РёС‚СЊ РєР°СЂС‚РѕС‡РєСѓ. РњР°СЂРєРµСЂС‹: "СѓРґР°Р»Рё", "СЃРѕС‚СЂРё", "СѓР±РµСЂРё РєР°СЂС‚РѕС‡РєСѓ".

6) move_card вЂ” РїРµСЂРµРјРµСЃС‚РёС‚СЊ РІ РґСЂСѓРіСѓСЋ РєРѕР»РѕРЅРєСѓ Р±РµР· Р·Р°РІРµСЂС€РµРЅРёСЏ. РњР°СЂРєРµСЂС‹: "РїРµСЂРµРЅРµСЃРё РІ doing", "РІ СЂР°Р±РѕС‚Сѓ".

7) add_checklist_items вЂ” РґРѕР±Р°РІРёС‚СЊ РїСѓРЅРєС‚С‹ РІ С‡РµРєР»РёСЃС‚ РЈР–Р• СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµР№ РєР°СЂС‚РѕС‡РєРё.
   РњР°СЂРєРµСЂС‹: "РґРѕР±Р°РІСЊ Рє РєР°СЂС‚РѕС‡РєРµ РҐ РїСѓРЅРєС‚ Y", "РІ С‡РµРєР»РёСЃС‚ РҐ РґРѕР±Р°РІСЊ Y" (РµСЃС‚СЊ РѕС‚СЃС‹Р»РєР° Рє СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµР№ РєР°СЂС‚РѕС‡РєРµ).

8) update_checklist_item вЂ” РїРµСЂРµРёРјРµРЅРѕРІР°С‚СЊ РєРѕРЅРєСЂРµС‚РЅС‹Р№ РїСѓРЅРєС‚ С‡РµРєР»РёСЃС‚Р°.

9) add_comment вЂ” РґРѕР±Р°РІРёС‚СЊ РєРѕРјРјРµРЅС‚Р°СЂРёР№ Рє РєР°СЂС‚РѕС‡РєРµ. РњР°СЂРєРµСЂС‹: "РїСЂРѕРєРѕРјРјРµРЅС‚РёСЂСѓР№", "РґРѕР±Р°РІСЊ РєРѕРјРјРµРЅС‚".

10-13) add_card_label / remove_card_label / add_card_member / remove_card_member.

14) attach_file вЂ” РїСЂРёРєСЂРµРїРёС‚СЊ С„Р°Р№Р»/СЃСЃС‹Р»РєСѓ Рє РєР°СЂС‚РѕС‡РєРµ.

15) set_persona вЂ” РїРµСЂРµРєР»СЋС‡РёС‚СЊ СЂРµР¶РёРј: "Р±СѓРґСЊ РєР°Рє РјР°РјР°/РёР»РѕРЅ/РґР·РµРЅ", "РїРµСЂРµРєР»СЋС‡РёСЃСЊ РЅР° zen".

16) link_board вЂ” РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РїСЂРёРІСЏР·С‹РІР°РµС‚ РґРѕСЃРєСѓ Trello (РїСЂРёСЃС‹Р»Р°РµС‚ СЃСЃС‹Р»РєСѓ https://trello.com/b/...).

17) chitchat вЂ” РЅРµС‚ РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ РґРµР№СЃС‚РІРёСЏ. РџСЂРёРІРµС‚СЃС‚РІРёРµ, Р±Р»Р°РіРѕРґР°СЂРЅРѕСЃС‚СЊ, РѕР±С‰РёР№ РІРѕРїСЂРѕСЃ.

РџР РђР’РР›Рђ:
- РџСЂРё СЃРѕРјРЅРµРЅРёРё РјРµР¶РґСѓ create_card Рё complete_task_from_text СЃРјРѕС‚СЂРё РЅР° Р’Р Р•РњРЇ Р“Р›РђР“РћР›Рђ:
  Р±СѓРґСѓС‰РµРµ/РёРЅС„РёРЅРёС‚РёРІ = create_card; РїСЂРѕС€РµРґС€РµРµ = complete_task_from_text.
- РџСЂРё СЃРѕРјРЅРµРЅРёРё РјРµР¶РґСѓ complete_task_from_text (РїСѓРЅРєС‚) Рё complete_card_by_text (РІСЃСЏ РєР°СЂС‚РѕС‡РєР°):
  РµСЃР»Рё СѓРїРѕРјСЏРЅСѓС‚Рѕ РЅР°Р·РІР°РЅРёРµ РєР°СЂС‚РѕС‡РєРё С†РµР»РёРєРѕРј Рё СЃР»РѕРІР° "Р·Р°РєСЂС‹С‚СЊ/Р·Р°РІРµСЂС€РёС‚СЊ/РіРѕС‚РѕРІР°" вЂ” СЌС‚Рѕ complete_card_by_text.
- response_text вЂ” РєРѕСЂРѕС‚РєР°СЏ (1 С„СЂР°Р·Р°) СЂРµРїР»РёРєР° РїРѕР»СЊР·РѕРІР°С‚РµР»СЋ СЃ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёРµРј.
"""


# ============================================================================
# 2) EXTRACTOR PROMPTS вЂ” СѓР·РєРёРµ РїСЂРѕРјРїС‚С‹ РїРѕРґ РєР°Р¶РґС‹Р№ intent
# ============================================================================

EXTRACTOR_PROMPTS: dict[str, str] = {
    "create_card": """\
РР·РІР»РµРєРё РїРѕР»СЏ РґР»СЏ СЃРѕР·РґР°РЅРёСЏ РќРћР’РћР™ РєР°СЂС‚РѕС‡РєРё РІ Trello.
Р’РµСЂРЅРё JSON: {"action_type":"create_card","card_name":"...","due":"...","start":null,"position":null,"checklist_name":"...","checklist_item":[...],"card_description":null,"response_text":"..."}.
РџСЂР°РІРёР»Р°:
- card_name вЂ” РєРѕСЂРѕС‚РєРѕРµ С‘РјРєРѕРµ РЅР°Р·РІР°РЅРёРµ Р·Р°РґР°С‡Рё (Р±РµР· РІСЂРµРјРµРЅРё Рё Р±РµР· РІСЃРїРѕРјРѕРіР°С‚РµР»СЊРЅС‹С… СЃР»РѕРІ "РЅРµ Р·Р°Р±С‹С‚СЊ/РЅСѓР¶РЅРѕ/РґР°РІР°Р№ РїРѕСЃС‚Р°РІРёРј").
- due вЂ” ISO 8601 (РЅР°РїСЂРёРјРµСЂ 2026-05-15T09:00:00.000Z). Р•СЃР»Рё РЅРµС‚ РћР”РќРћР—РќРђР§РќРћР“Рћ СЃСЂРѕРєР° вЂ” РїРѕСЃС‚Р°РІСЊ null (СЃРµСЂРІРµСЂ СѓС‚РѕС‡РЅРёС‚).
- Р•СЃР»Рё РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РїРµСЂРµС‡РёСЃР»СЏРµС‚ РїРѕРґР·Р°РґР°С‡Рё РІ С‚РѕР№ Р¶Рµ С„СЂР°Р·Рµ вЂ” checklist_item СЌС‚Рѕ РњРђРЎРЎРР’ СЃС‚СЂРѕРє (РєРѕСЂРѕС‚РєРёРµ РїСѓРЅРєС‚С‹), checklist_name = "РЁР°РіРё".
- РќРёРєРѕРіРґР° РЅРµ РІС‹РґСѓРјС‹РІР°Р№ РґР°С‚Сѓ.
""",
    "update_card": """\
РР·РІР»РµРєРё РїРѕР»СЏ РґР»СЏ РѕР±РЅРѕРІР»РµРЅРёСЏ СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµР№ РєР°СЂС‚РѕС‡РєРё.
Р’РµСЂРЅРё JSON: {"action_type":"update_card","card_id":null,"card_name":null,"card_description":null,"due":null,"start":null,"due_complete":null,"closed":null,"response_text":"..."}.
Р•СЃР»Рё РІ СЂРµРїР»РёРєРµ СЏРІРЅРѕ СѓРєР°Р·Р°РЅРѕ, РєР°РєРѕРµ РїРѕР»Рµ РјРµРЅСЏС‚СЊ вЂ” Р·Р°РїРѕР»РЅРё С‚РѕР»СЊРєРѕ РµРіРѕ. РЎСЂРѕРє С‚РѕР»СЊРєРѕ РІ ISO 8601.
""",
    "delete_card": """\
Р’РµСЂРЅРё JSON: {"action_type":"delete_card","card_id":null,"card_name":null,"response_text":"..."}.
Р•СЃР»Рё РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РЅР°Р·РІР°Р» РєР°СЂС‚РѕС‡РєСѓ вЂ” Р·Р°РЅРµСЃРё РµС‘ РІ card_name (РєРѕСЂРѕС‚РєРѕРµ РЅР°Р·РІР°РЅРёРµ).
""",
    "move_card": """\
Р’РµСЂРЅРё JSON: {"action_type":"move_card","card_id":null,"card_name":null,"list_name":"inbox|doing|done","response_text":"..."}.
list_name вЂ” С†РµР»РµРІР°СЏ РєРѕР»РѕРЅРєР° РёР· {inbox, doing, done}.
""",
    "complete_card_by_text": """\
РР·РІР»РµРєРё РЅР°Р·РІР°РЅРёРµ РєР°СЂС‚РѕС‡РєРё, РєРѕС‚РѕСЂСѓСЋ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ С…РѕС‡РµС‚ Р—РђРљР Р«РўР¬/РџР•Р Р•Р’Р•РЎРўР Р’ Р—РђР’Р•Р РЁРЃРќРќР«Р•.
Р’РµСЂРЅРё JSON: {"action_type":"complete_card_by_text","match_text":"...","response_text":"..."}.
match_text вЂ” РєРѕСЂРѕС‚РєРѕРµ РЅР°Р·РІР°РЅРёРµ РєР°СЂС‚РѕС‡РєРё Р‘Р•Р— СЃР»РѕРІ "Р·Р°РєСЂРѕР№/РїРµСЂРµРІРµСЃС‚Рё/Р·Р°РІРµСЂС€РёС‚СЊ/РєР°СЂС‚РѕС‡РєР°/Р·Р°РґР°С‡Р°".
РџСЂРёРјРµСЂ: "РњР°С€РёРЅСѓ РѕР±СЃР»СѓР¶РёР»Рё, РјРѕР¶РЅРѕ Р·Р°РєСЂС‹РІР°С‚СЊ СЌС‚Сѓ Р·Р°РґР°С‡Сѓ" в†’ match_text = "РѕР±СЃР»СѓР¶РёРІР°РЅРёРµ РјР°С€РёРЅС‹".
РџСЂРёРјРµСЂ: "Р—Р°РєСЂРѕР№ РєР°СЂС‚РѕС‡РєСѓ РїРѕСЃР°РґРёС‚СЊ РґРµСЂРµРІРѕ" в†’ match_text = "РїРѕСЃР°РґРёС‚СЊ РґРµСЂРµРІРѕ".
РќРµ РїРѕРґСЃС‚Р°РІР»СЏР№ card_id, СЃРµСЂРІРµСЂ РЅР°Р№РґС‘С‚ РєР°СЂС‚РѕС‡РєСѓ РїРѕ РЅР°Р·РІР°РЅРёСЋ.
""",
    "add_checklist_items": """\
РР·РІР»РµРєРё РїСѓРЅРєС‚С‹ РґР»СЏ РґРѕР±Р°РІР»РµРЅРёСЏ РІ С‡РµРєР»РёСЃС‚ РЈР–Р• СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµР№ РєР°СЂС‚РѕС‡РєРё.
Р’РµСЂРЅРё JSON: {"action_type":"create_checklist_item","card_id":null,"card_name":null,"checklist_name":"РЁР°РіРё","checklist_item":"...","response_text":"..."}.
- checklist_item вЂ” СЃС‚СЂРѕРєР° РёР»Рё РЅРµСЃРєРѕР»СЊРєРѕ РїСѓРЅРєС‚РѕРІ С‡РµСЂРµР· "; " / РїРµСЂРµРІРѕРґ СЃС‚СЂРѕРєРё (С‚РѕР»СЊРєРѕ С‚Рѕ, С‡С‚Рѕ РґРѕР±Р°РІР»СЏРµРј РІ С‡РµРєР»РёСЃС‚).
- card_name вЂ” РўРћР›Р¬РљРћ РєРѕСЂРѕС‚РєРѕРµ РЅР°Р·РІР°РЅРёРµ СЃР°РјРѕР№ РєР°СЂС‚РѕС‡РєРё РЅР° РґРѕСЃРєРµ. РќРёРєРѕРіРґР° РЅРµ РїРѕРјРµС‰Р°Р№ СЃСЋРґР° РїСЂРµРґРјРµС‚С‹/РїСѓРЅРєС‚С‹
  РёР· checklist_item (РЅР°РїСЂРёРјРµСЂ В«Р°РЅРіР»РёР№СЃРєРёР№В», В«РіРµРѕРіСЂР°С„РёСЏВ»). Р•СЃР»Рё РЅР°Р·РІР°РЅРёРµ РґР°РЅРѕ РІ РєР°РІС‹С‡РєР°С…
  (В«РЎРґРµР»Р°С‚СЊ СѓСЂРѕРєРёВ») вЂ” СЌС‚Рѕ РЅР°Р·РІР°РЅРёРµ РєР°СЂС‚РѕС‡РєРё, РµРіРѕ РјРѕР¶РЅРѕ РїСЂРѕРґСѓР±Р»РёСЂРѕРІР°С‚СЊ РІ card_name РёР»Рё РѕСЃС‚Р°РІРёС‚СЊ null.
- Р•СЃР»Рё РµСЃС‚СЊ СЃСЃС‹Р»РєР° trello.com/c/... РёР»Рё 24-СЃРёРјРІРѕР»СЊРЅС‹Р№ id вЂ” Р·Р°РЅРµСЃРё РІ card_id.
- Р•СЃР»Рё РЅРµС‚ РЅРё id, РЅРё РЅР°Р·РІР°РЅРёСЏ вЂ” card_id Рё card_name null; СЃРµСЂРІРµСЂ РёР·РІР»РµС‡С‘С‚ РЅР°Р·РІР°РЅРёРµ РёР· РєР°РІС‹С‡РµРє РІ С‚РµРєСЃС‚Рµ.
""",
    "complete_task_from_text": """\
РР·РІР»РµРєРё РєРѕСЂРѕС‚РєРѕРµ РѕРїРёСЃР°РЅРёРµ С‚РѕРіРѕ, С‡С‚Рѕ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РЈР–Р• Р’Р«РџРћР›РќРР› (РѕРґРёРЅ РїСѓРЅРєС‚ С‡РµРєР»РёСЃС‚Р°).
Р’РµСЂРЅРё JSON: {"action_type":"complete_task_from_text","match_text":"...","response_text":"..."}.
match_text вЂ” Р±РµР· РјРµСЃС‚РѕРёРјРµРЅРёР№ Рё РІСЃРїРѕРјРѕРіР°С‚РµР»СЊРЅС‹С… СЃР»РѕРІ: "СЏ РїРѕРјС‹Р» Р»РѕР±РѕРІРѕРµ СЃС‚РµРєР»Рѕ" в†’ "РїРѕРјС‹Р» Р»РѕР±РѕРІРѕРµ СЃС‚РµРєР»Рѕ".
РќРµ РІС‹РґСѓРјС‹РІР°Р№ card_id Рё checklist_id вЂ” СЃРµСЂРІРµСЂ РЅР°Р№РґС‘С‚ РїСѓРЅРєС‚ РїРѕ С‚РµРєСЃС‚Сѓ.
""",
    "update_checklist_item": """\
Р’РµСЂРЅРё JSON: {"action_type":"update_checklist_item","card_id":null,"checklist_id":null,"check_item_id":null,"check_item_complete":null,"check_item_new_name":null,"response_text":"..."}.
Р—Р°РїРѕР»РЅРё С‚РѕР»СЊРєРѕ СѓРєР°Р·Р°РЅРЅС‹Рµ РїРѕР»СЊР·РѕРІР°С‚РµР»РµРј РїРѕР»СЏ.
""",
    "add_comment": """\
Р’РµСЂРЅРё JSON: {"action_type":"add_comment","card_id":null,"card_name":null,"comment_text":"...","response_text":"..."}.
""",
    "add_card_label": """\
Р’РµСЂРЅРё JSON: {"action_type":"add_card_label","card_id":null,"label_id":null,"response_text":"..."}.
""",
    "remove_card_label": """\
Р’РµСЂРЅРё JSON: {"action_type":"remove_card_label","card_id":null,"label_id":null,"response_text":"..."}.
""",
    "add_card_member": """\
Р’РµСЂРЅРё JSON: {"action_type":"add_card_member","card_id":null,"member_id":null,"response_text":"..."}.
""",
    "remove_card_member": """\
Р’РµСЂРЅРё JSON: {"action_type":"remove_card_member","card_id":null,"member_id":null,"response_text":"..."}.
""",
    "attach_file": """\
Р’РµСЂРЅРё JSON: {"action_type":"attach_file","card_id":null,"card_name":null,"file_url":null,"response_text":"..."}.
""",
    "set_category": """\
Верни JSON: {"action_type":"set_category","card_id":null,"card_name":null,"category_key":"home|work|interests|routine|study|health","response_text":"..."}.
Если карточка не указана — card_id/card_name оставь null.
""",
    "create_routine_template": """\
Верни JSON: {"action_type":"create_routine_template","routine_name":"...","category_key":null,"routine_duration_min":null,"routine_cron":null,"checklist_item":null,"response_text":"..."}.
routine_duration_min — минуты (целое), если указано явно.
checklist_item — пункты через ";" если пользователь их перечислил.
""",
    "pause_routine_template": """\
Верни JSON: {"action_type":"pause_routine_template","routine_name":"...","response_text":"..."}.
""",
    "resume_routine_template": """\
Верни JSON: {"action_type":"resume_routine_template","routine_name":"...","response_text":"..."}.
""",
    "list_routine_templates": """\
Верни JSON: {"action_type":"list_routine_templates","response_text":"..."}.
""",
    "set_persona": """\
Р’РµСЂРЅРё JSON: {"action_type":"set_persona","metadata":{"persona":"mom|elon|zen"},"response_text":"..."}.
""",
    "link_board": """\
Р’РµСЂРЅРё JSON: {"action_type":"link_board","metadata":{"board_url":"https://trello.com/b/..."},"response_text":"..."}.
""",
}


# ============================================================================
# 3) CARD SEARCH AGENT вЂ” СЃРµРјР°РЅС‚РёС‡РµСЃРєРёР№ РїРѕРёСЃРє РєР°СЂС‚РѕС‡РµРє РїРѕ СЂРµРїР»РёРєРµ
# ============================================================================

CARD_SEARCH_PROMPT = """\
РўС‹ вЂ” СЃРµРјР°РЅС‚РёС‡РµСЃРєРёР№ РїРѕРёСЃРєРѕРІРёРє РїРѕ РєР°СЂС‚РѕС‡РєР°Рј Trello.

РќР° РІС…РѕРґ РїРѕР»СѓС‡Р°РµС€СЊ JSON {"query": "...", "cards": [{"id":"...","name":"...","incomplete_items":[...],"total_items":N}]}.
- query вЂ” СЂРµРїР»РёРєР° РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ РІ СЃРІРѕР±РѕРґРЅРѕР№ С„РѕСЂРјРµ (СЂСѓСЃСЃРєРёР№, СЂР°Р·РіРѕРІРѕСЂРЅС‹Р№).
- cards вЂ” РѕС‚РєСЂС‹С‚С‹Рµ РєР°СЂС‚РѕС‡РєРё РЅР° РґРѕСЃРєРµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ.

Р—Р°РґР°С‡Р°: РІРµСЂРЅСѓС‚СЊ JSON {"hits":[{"card_id":"...","score":0.0..1.0,"reason":"РєРѕСЂРѕС‚РєРѕ РїРѕС‡РµРјСѓ"}]}.

РџР РђР’РР›Рђ:
- Р’РєР»СЋС‡Р°Р№ РўРћР›Р¬РљРћ С‚Рµ РєР°СЂС‚РѕС‡РєРё, Рє РєРѕС‚РѕСЂС‹Рј СЂРµРїР»РёРєР° СЂРµР°Р»СЊРЅРѕ РѕС‚РЅРѕСЃРёС‚СЃСЏ.
- РСЃРїРѕР»СЊР·СѓР№ РўРћР›Р¬РљРћ id РёР· РїРµСЂРµРґР°РЅРЅРѕРіРѕ СЃРїРёСЃРєР°. РќРёРєРѕРіРґР° РЅРµ РІС‹РґСѓРјС‹РІР°Р№ id.
- РЎРѕСЂС‚РёСЂСѓР№ РїРѕ score СѓР±С‹РІР°РЅРёСЋ.
- score >= 0.7 вЂ” С‚РѕС‡РЅС‹Р№/СѓРІРµСЂРµРЅРЅС‹Р№ РјР°С‚С‡; 0.4..0.7 вЂ” РІРµСЂРѕСЏС‚РЅС‹Р№ (С‚РµРјР°С‚РёРєР° СЃРѕРІРїР°РґР°РµС‚); < 0.4 вЂ” РќР• РІРѕР·РІСЂР°С‰Р°Р№.
- Р•СЃР»Рё РЅРёС‡РµРіРѕ РЅРµ РїРѕРґС…РѕРґРёС‚ вЂ” РІРµСЂРЅРё {"hits":[]}.

РџР РРњР•Р Р«:
1) query: "РјР°С€РёРЅСѓ РѕР±СЃР»СѓР¶РёР»Рё, РјРѕР¶РЅРѕ Р·Р°РєСЂС‹РІР°С‚СЊ"
   cards: [{"id":"abc","name":"РћР±СЃР»СѓР¶РёРІР°РЅРёРµ РјР°С€РёРЅС‹","incomplete_items":["РџРѕРјС‹С‚СЊ Р»РѕР±РѕРІРѕРµ"]},{"id":"def","name":"РљСѓРїРёС‚СЊ С…Р»РµР±"}]
   РѕС‚РІРµС‚: {"hits":[{"card_id":"abc","score":0.92,"reason":"name РїСЂСЏРјРѕ РѕРїРёСЃС‹РІР°РµС‚ РѕР±СЃР»СѓР¶РёРІР°РЅРёРµ РјР°С€РёРЅС‹"}]}
2) query: "РѕС‚С‡С‘С‚ РїРѕ РїСЂРѕРµРєС‚Сѓ Рђ РіРѕС‚РѕРІ"
   cards: [{"id":"a","name":"РћС‚С‡С‘С‚ РїРѕ РїСЂРѕРµРєС‚Сѓ Рђ"},{"id":"b","name":"РћС‚С‡С‘С‚ РїРѕ РїСЂРѕРµРєС‚Сѓ Р‘"}]
   РѕС‚РІРµС‚: {"hits":[{"card_id":"a","score":0.95,"reason":"СЏРІРЅРѕРµ СѓРїРѕРјРёРЅР°РЅРёРµ РїСЂРѕРµРєС‚Р° Рђ"}]}
3) query: "Р·Р°РєСЂРѕР№ С‡С‚Рѕ-С‚Рѕ РїСЂРѕ РјР°С€РёРЅСѓ"
   cards: [{"id":"abc","name":"РћР±СЃР»СѓР¶РёРІР°РЅРёРµ РјР°С€РёРЅС‹"},{"id":"xyz","name":"РљСѓРїРёС‚СЊ РјР°С€РёРЅСѓ"}]
   РѕС‚РІРµС‚: {"hits":[{"card_id":"abc","score":0.6,"reason":"РјР°С€РёРЅР°"},{"id":"xyz","score":0.55,"reason":"РјР°С€РёРЅР°"}]}
"""


# ============================================================================
# РќРѕСЂРјР°Р»РёР·Р°С‚РѕСЂС‹ JSON-payload (Р·Р°С‰РёС‚Р° РѕС‚ С‚РёРїРёС‡РЅС‹С… В«СЃС‚СЂР°РЅРЅРѕСЃС‚РµР№В» LLM)
# ============================================================================


def _normalize_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    rt = out.get("response_text")
    if rt is None or (isinstance(rt, str) and not rt.strip()):
        out["response_text"] = "РџСЂРёРЅСЏС‚Рѕ, СЂР°Р±РѕС‚Р°РµРј."
    elif not isinstance(rt, str):
        out["response_text"] = str(rt).strip() or "РџСЂРёРЅСЏС‚Рѕ, СЂР°Р±РѕС‚Р°РµРј."
    if out.get("metadata") is None:
        out["metadata"] = {}
    ci = out.get("checklist_item")
    if isinstance(ci, list):
        parts = [str(x).strip() for x in ci if x is not None and str(x).strip()]
        out["checklist_item"] = "; ".join(parts) if parts else None
    return out


def _normalize_router_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    intent = out.get("intent")
    if intent not in VALID_INTENTS:
        out["intent"] = "chitchat"
    if not isinstance(out.get("reasoning"), str):
        out["reasoning"] = ""
    rt = out.get("response_text")
    if rt is None or not isinstance(rt, str) or not rt.strip():
        out["response_text"] = "РџСЂРёРЅСЏС‚Рѕ, СЂР°Р±РѕС‚Р°РµРј."
    return out


def _render_intent_context_block(intent_context: str | None) -> str:
    if not intent_context:
        return ""
    trimmed = intent_context.strip()
    if not trimmed:
        return ""
    if len(trimmed) > 1800:
        trimmed = trimmed[-1800:]
    return (
        "\n\n[Intent History Context]\n"
        f"{trimmed}\n"
        "Use this context as a hint for disambiguation (create/update/complete). "
        "Never invent card ids and never execute older commands automatically."
    )


# ============================================================================
# Service
# ============================================================================


class AgentService:
    """LLM-Р°РіРµРЅС‚. Р”РІСѓС…С€Р°РіРѕРІС‹Р№: СЃРЅР°С‡Р°Р»Р° classify_intent, РїРѕС‚РѕРј extract_fields."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def infer_action(
        self,
        text: str,
        persona: str = "mom",
        *,
        forced_intent: str | None = None,
        intent_context: str | None = None,
    ) -> AgentResult:
        if forced_intent and forced_intent in VALID_INTENTS:
            decision = IntentDecision(
                intent=forced_intent,  # type: ignore[arg-type]
                reasoning="forced_by_orchestrator",
                response_text="РџСЂРёРЅСЏС‚Рѕ.",
            )
            logger.info("intent_router: FORCED intent=%s text=%r", forced_intent, text)
        else:
            # РЁР°Рі 0: РґРµС€С‘РІС‹Р№ rule-based pre-router. Р•СЃР»Рё РєР»СЋС‡РµРІС‹Рµ РјР°СЂРєРµСЂС‹
            # РѕРґРЅРѕР·РЅР°С‡РЅРѕ СѓРєР°Р·С‹РІР°СЋС‚ РЅР° РёРЅС‚РµРЅС‚ вЂ” РЅРµ РїР»Р°С‚РёРј Р·Р° LLM.
            rule_intent = quick_classify_intent(text)
            if rule_intent is not None:
                decision = IntentDecision(
                    intent=rule_intent,  # type: ignore[arg-type]
                    reasoning=f"rule_based:{rule_intent}",
                    response_text="РџСЂРёРЅСЏС‚Рѕ, СЂР°Р±РѕС‚Р°РµРј.",
                )
                logger.info(
                    "intent_router: RULE-BASED intent=%s text=%r",
                    rule_intent,
                    text,
                )
            else:
                # РЁР°Рі 1: LLM-СЂРѕСѓС‚РµСЂ РґР»СЏ РІСЃРµС… РЅРµРѕРґРЅРѕР·РЅР°С‡РЅС‹С… СЃР»СѓС‡Р°РµРІ.
                decision = await self._classify_intent(text, persona, intent_context=intent_context)
                logger.info(
                    "intent_router: LLM intent=%s reasoning=%r text=%r",
                    decision.intent,
                    decision.reasoning,
                    text,
                )

        if decision.intent == "chitchat":
            return AgentResult(
                action=AgentAction(action_type="none", reasoning=decision.reasoning),
                response_text=decision.response_text,
            )

        action = await self._extract_fields(text, persona, decision, intent_context=intent_context)
        logger.info(
            "intent_extractor: action_type=%s card_name=%r match_text=%r due=%r",
            action.action_type,
            action.card_name,
            action.match_text,
            action.due,
        )
        return AgentResult(action=action, response_text=decision.response_text)

    # ------------------------------------------------- semantic card search
    async def search_cards_semantic(
        self,
        query: str,
        cards_payload: list[dict[str, Any]],
        persona: str = "mom",
    ) -> list[CardSearchHit]:
        """LLM-СЃРµРјР°РЅС‚РёС‡РµСЃРєРёР№ РїРѕРёСЃРє РєР°СЂС‚РѕС‡РµРє РїРѕ С‚РµРєСЃС‚Сѓ.

        cards_payload вЂ” РєРѕРјРїР°РєС‚РЅС‹Рµ СЃРІРѕРґРєРё РѕС‚ `card_search.format_cards_for_search`.
        Р’РѕР·РІСЂР°С‰Р°РµС‚ СЃРїРёСЃРѕРє `CardSearchHit` (С‚РѕР»СЊРєРѕ РІР°Р»РёРґРЅС‹Рµ id, score >= 0.4).
        Р­С‚РѕС‚ РјРµС‚РѕРґ РќР• РѕР±СЂР°С‰Р°РµС‚СЃСЏ Рє Trello вЂ” СЃРїРёСЃРѕРє РєР°СЂС‚РѕС‡РµРє РїРѕРґР°С‘С‚СЃСЏ СЃРЅР°СЂСѓР¶Рё.
        """
        if not query.strip() or not cards_payload:
            return []

        user_msg = json.dumps(
            {"query": query, "cards": cards_payload}, ensure_ascii=False,
        )
        system = (
            f"{PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS['mom'])}\n\n"
            f"{CARD_SEARCH_PROMPT}"
        )
        try:
            raw = await self._call_llm_json(system, user_msg, step="card_search")
        except OpenRouterLLMError as exc:
            logger.warning("card_search LLM failed: %s", exc)
            return []

        hits_raw = raw.get("hits") if isinstance(raw, dict) else None
        if not isinstance(hits_raw, list):
            logger.warning("card_search: payload Р±РµР· РјР°СЃСЃРёРІР° hits: %s", raw)
            return []

        valid_ids = {str(c.get("id")): c for c in cards_payload if c.get("id")}
        out: list[CardSearchHit] = []
        for h in hits_raw:
            if not isinstance(h, dict):
                continue
            cid = str(h.get("card_id") or "").strip()
            if cid not in valid_ids:
                continue
            try:
                score = float(h.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(1.0, score))
            if score < 0.4:
                continue
            card = valid_ids[cid]
            out.append(
                CardSearchHit(
                    card_id=cid,
                    card_name=str(card.get("name", "")),
                    score=score,
                    reason=str(h.get("reason") or ""),
                    incomplete_items=tuple(card.get("incomplete_items") or []),
                    total_items=int(card.get("total_items") or 0),
                    list_id=None,
                )
            )
        out.sort(key=lambda h: h.score, reverse=True)
        logger.info(
            "card_search: query=%r hits=%s",
            query,
            [(h.card_id, round(h.score, 2), h.card_name) for h in out],
        )
        return out

    # ------------------------------------------------------------------ step 1
    async def _classify_intent(
        self,
        text: str,
        persona: str,
        *,
        intent_context: str | None = None,
    ) -> IntentDecision:
        system = (
            f"{PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS['mom'])}\n"
            f"{calendar_context_for_prompt()}\n\n"
            f"{ROUTER_PROMPT}"
            f"{_render_intent_context_block(intent_context)}"
        )
        raw_payload = await self._call_llm_json(system, text, step="router")
        raw_payload = _normalize_router_payload(raw_payload)
        try:
            return IntentDecision.model_validate(raw_payload)
        except ValidationError as exc:
            logger.warning("intent_router validation failed: %s payload=%s", exc, raw_payload)
            return IntentDecision(intent="chitchat", reasoning="router_validation_failed")

    # ------------------------------------------------------------------ step 2
    async def _extract_fields(
        self,
        text: str,
        persona: str,
        decision: IntentDecision,
        *,
        intent_context: str | None = None,
    ) -> AgentAction:
        intent = decision.intent
        extractor_prompt = EXTRACTOR_PROMPTS.get(intent)
        if extractor_prompt is None:
            logger.warning("no extractor for intent=%s, returning none", intent)
            return AgentAction(action_type="none", reasoning=decision.reasoning)

        system = (
            f"{PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS['mom'])}\n"
            f"{calendar_context_for_prompt()}\n\n"
            f"РЈР¶Рµ РѕРїСЂРµРґРµР»РµРЅРѕ РЅР°РјРµСЂРµРЅРёРµ: {intent}.\n"
            f"Р›РѕРіРёРєР° СЂРѕСѓС‚РµСЂР°: {decision.reasoning}\n\n"
            f"{extractor_prompt}"
            f"{_render_intent_context_block(intent_context)}"
        )
        raw_payload = await self._call_llm_json(system, text, step=f"extractor:{intent}")
        raw_payload = _normalize_action_payload(raw_payload)
        # РЈР±РµРґРёРјСЃСЏ, С‡С‚Рѕ action_type РІРЅСѓС‚СЂРё extractor-payload РЅРµ СЃСЉРµС…Р°Р».
        if intent == "add_checklist_items":
            raw_payload["action_type"] = "create_checklist_item"
        else:
            raw_payload["action_type"] = intent
        # РџРµСЂРµРЅРµСЃС‘Рј reasoning РёР· СЂРѕСѓС‚РµСЂР°, С‡С‚РѕР±С‹ Р±С‹Р»Рѕ РІРёРґРЅРѕ РІ Р»РѕРіР°С… РѕСЂРєРµСЃС‚СЂР°С‚РѕСЂР°.
        raw_payload.setdefault("reasoning", decision.reasoning)

        try:
            return AgentAction.model_validate(raw_payload)
        except ValidationError as exc:
            logger.warning(
                "intent_extractor validation failed: intent=%s err=%s payload=%s",
                intent,
                exc,
                raw_payload,
            )
            raise OpenRouterLLMError(
                "РњРѕРґРµР»СЊ РІРµСЂРЅСѓР»Р° РїРѕР»СЏ РІ РЅРµРѕР¶РёРґР°РЅРЅРѕРј С„РѕСЂРјР°С‚Рµ. РџРѕРїСЂРѕР±СѓР№С‚Рµ РєРѕСЂРѕС‡Рµ РёР»Рё СЏРІРЅРѕ СѓРєР°Р¶РёС‚Рµ РєР°СЂС‚РѕС‡РєСѓ.",
            ) from exc

    # ---------------------------------------------------------------- LLM call
    async def _call_llm_json(self, system: str, user: str, *, step: str) -> dict[str, Any]:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            raise OpenRouterLLMError(
                "РњРѕРґРµР»СЊ (OpenRouter/OpenAI): РЅРµРІРµСЂРЅС‹Р№ РёР»Рё РїСѓСЃС‚РѕР№ РєР»СЋС‡. РџСЂРѕРІРµСЂСЊС‚Рµ OPENROUTER_API_KEY РёР»Рё OPENAI_API_KEY.",
            ) from exc
        except RateLimitError as exc:
            raise OpenRouterLLMError(
                "РњРѕРґРµР»СЊ: СЃР»РёС€РєРѕРј РјРЅРѕРіРѕ Р·Р°РїСЂРѕСЃРѕРІ. РџРѕРґРѕР¶РґРёС‚Рµ Рё РїРѕРІС‚РѕСЂРёС‚Рµ.",
            ) from exc
        except APIConnectionError as exc:
            raise OpenRouterLLMError(
                "РњРѕРґРµР»СЊ: РЅРµС‚ СЃРѕРµРґРёРЅРµРЅРёСЏ СЃ OpenRouter/OpenAI (СЃРµС‚СЊ, VPN, OPENROUTER_BASE_URL).",
            ) from exc
        except APIStatusError as exc:
            raise OpenRouterLLMError(
                f"РњРѕРґРµР»СЊ: РѕС€РёР±РєР° API ({exc.status_code}). РџСЂРѕРІРµСЂСЊС‚Рµ CHAT_MODEL Рё РєР»СЋС‡.",
            ) from exc

        raw = completion.choices[0].message.content or "{}"
        logger.debug("LLM raw response (%s): %s", step, raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenRouterLLMError(
                "РњРѕРґРµР»СЊ РІРµСЂРЅСѓР»Р° РЅРµ JSON. РЈС‚РѕС‡РЅРёС‚Рµ Р·Р°РїСЂРѕСЃ РёР»Рё СЃРјРµРЅРёС‚Рµ CHAT_MODEL.",
            ) from exc
        if not isinstance(payload, dict):
            raise OpenRouterLLMError("РњРѕРґРµР»СЊ РІРµСЂРЅСѓР»Р° РЅРµ РѕР±СЉРµРєС‚ JSON.")
        return payload




