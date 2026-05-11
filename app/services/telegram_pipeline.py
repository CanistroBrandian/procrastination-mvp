from __future__ import annotations

import json
import logging
import tempfile
from collections import OrderedDict
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import (
    AppError,
    BoardNotLinkedError,
    OpenAITranscriptionError,
    OpenRouterLLMError,
    OpenRouterTranscriptionError,
    OrchestratorValidationError,
    TelegramAPIError,
    TrelloAPIError,
)
from app.db.models import Persona
from app.db.repositories import (
    ClarificationRepository,
    IntentHistoryRepository,
    UserProfileRepository,
)
from app.integrations.telegram import TelegramClient
from app.integrations.trello import TrelloClient
from app.services.active_cards import (
    build_cards_overview_messages,
    build_cards_by_days_window_messages,
    build_active_cards_messages,
    parse_cards_nl_query,
    parse_cards_command_arguments,
)
from app.services.agent import AgentService
from app.services.asr import build_transcription_service
from app.services.onboarding import OnboardingService
from app.services.orchestrator import TaskOrchestrator
from app.services.intent_rules import quick_classify_intent
from app.services.trello_board_filters import drop_cards_in_archived_lists

logger = logging.getLogger(__name__)

# Защита от повторной обработки одного update_id (несколько воркеров / повтор Telegram).
_MAX_SEEN_UPDATES = 4000
_seen_update_ids: OrderedDict[int, None] = OrderedDict()


def _already_processed_update(update_id: int) -> bool:
    if update_id in _seen_update_ids:
        return True
    _seen_update_ids[update_id] = None
    while len(_seen_update_ids) > _MAX_SEEN_UPDATES:
        _seen_update_ids.popitem(last=False)
    return False


def _get_text(update: dict) -> str | None:
    msg = update.get("message", {})
    return msg.get("text") or msg.get("caption")


def _fallback_message(exc: BaseException) -> str:
    """Если ошибка не классифицирована — краткий текст без огромного traceback."""
    return f"Неожиданная ошибка: {exc!s}"[:500]


async def process_telegram_update(
    db: AsyncSession,
    update: dict,
    settings: Settings,
    openai_client: AsyncOpenAI,
) -> None:
    """Общая логика для webhook и long polling."""
    raw_uid = update.get("update_id")
    if raw_uid is not None:
        try:
            uid_int = int(raw_uid)
        except (TypeError, ValueError):
            uid_int = None
        if uid_int is not None and _already_processed_update(uid_int):
            return

    message = update.get("message") or {}
    if not message:
        return

    user_id = message["from"]["id"]
    chat_id = message["chat"]["id"]
    tg_client = TelegramClient(settings.telegram_bot_token)
    profile_repo = UserProfileRepository(db)
    clarification_repo = ClarificationRepository(db)
    intent_history_repo = IntentHistoryRepository(db)
    profile = await profile_repo.get_or_create(user_id)
    trello_client = TrelloClient(settings.trello_api_key, profile.trello_token or settings.trello_api_token)
    orchestrator = TaskOrchestrator(
        agent_service=AgentService(client=openai_client, model=settings.chat_model),
        trello_client=trello_client,
        clarification_repo=clarification_repo,
    )

    text = _get_text(update)
    if text and text.startswith("/start"):
        await tg_client.send_message(
            chat_id,
            "Привет. Шаги: 1) /boards — выбрать доску  2) /link <board_id>  3) пиши задачи текстом или голосом.\n"
            "Команды: /persona elon|zen|mom  ·  /cards — активные карточки (фильтр: /cards 7, сегодня, завтра, неделя, просрочка)\n\n"
            "Как устроен доступ: см. GET / (JSON) на сервере API — там webhook URL и заголовок секрета.",
        )
        return
    if text and text.strip() in ("/reset", "/cancel"):
        await clarification_repo.clear(profile.telegram_user_id)
        await tg_client.send_message(
            chat_id,
            "Контекст уточнения сброшен. Можете начинать с новой команды.",
        )
        return
    if text and text.startswith("/persona"):
        parts = text.split()
        if len(parts) != 2 or parts[1] not in {"elon", "zen", "mom"}:
            await tg_client.send_message(chat_id, "Используй: /persona elon|zen|mom")
            return
        profile.persona = Persona(parts[1])
        await profile_repo.save(profile)
        await tg_client.send_message(chat_id, f"Персона переключена: {parts[1]}")
        return
    if text and text.startswith("/link"):
        parts = text.split()
        board_id = parts[1] if len(parts) > 1 else settings.default_trello_board_id
        if not board_id:
            await tg_client.send_message(chat_id, "Передай board_id: /link <board_id>")
            return
        try:
            onboarding = OnboardingService(trello_client)
            await onboarding.link_default_board(profile, board_id)
            await profile_repo.save(profile)
            await tg_client.send_message(chat_id, "Доска подключена. Готов принимать задачи.")
        except TrelloAPIError as exc:
            await tg_client.send_message(chat_id, exc.user_message)
        return
    if text and text.startswith("/boards"):
        try:
            boards = await trello_client.list_boards()
        except TrelloAPIError as exc:
            await tg_client.send_message(chat_id, exc.user_message)
            return
        if not boards:
            await tg_client.send_message(chat_id, "Доступных досок не найдено.")
            return
        preview = "\n".join([f"- {b['name']}: `{b['id']}`" for b in boards[:10]])
        await tg_client.send_message(chat_id, f"Доски:\n{preview}\n\nПривязка: /link <board_id>")
        return

    pending = await clarification_repo.get(profile.telegram_user_id)

    if message.get("voice"):
        # Как в build_transcription_service: OpenRouter для Whisper или ключ OpenAI.
        or_key = (settings.openrouter_api_key or "").strip()
        oa_key = (settings.openai_api_key or "").strip()
        has_asr_key = bool(or_key or oa_key)
        if not has_asr_key:
            await tg_client.send_message(
                chat_id,
                "Голос не обработан: в .env нужен OPENROUTER_API_KEY (Whisper на OpenRouter) или OPENAI_API_KEY. "
                "Если распознавание всё равно иногда приходит — у вас, скорее всего, запущено несколько копий "
                "сервера: одна без ключей, другая с ключами. Оставьте один процесс uvicorn.",
            )
            return
        try:
            file_id = message["voice"]["file_id"]
            file_path = await tg_client.get_file_path(file_id)
            download_url = tg_client.build_file_download_url(file_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                async with httpx.AsyncClient(timeout=60) as client:
                    res = await client.get(download_url)
                    res.raise_for_status()
                    tmp.write(res.content)
                audio_path = Path(tmp.name)
            transcription = build_transcription_service(settings, openai_client)
            text = await transcription.transcribe(audio_path)
            await tg_client.send_message(chat_id, f"Распознал: {text}")
        except TelegramAPIError as exc:
            await tg_client.send_message(chat_id, exc.user_message)
            return
        except (OpenRouterTranscriptionError, OpenAITranscriptionError) as exc:
            await tg_client.send_message(chat_id, exc.user_message)
            return
        except httpx.HTTPError as exc:
            await tg_client.send_message(
                chat_id,
                f"Не удалось скачать голосовой файл с Telegram: {exc!s}",
            )
            return

    if not text:
        await tg_client.send_message(chat_id, "Не понял сообщение. Отправь текст или голосовое.")
        return

    if message.get("document"):
        try:
            file_id = message["document"]["file_id"]
            file_path = await tg_client.get_file_path(file_id)
            text = f"{text}\nfile_url={tg_client.build_file_download_url(file_path)}"
        except TelegramAPIError as exc:
            await tg_client.send_message(chat_id, exc.user_message)
            return

    # Слэш-команда /cards и NL-фильтр «какие задачи на сегодня?» проверяем ПОСЛЕ ASR,
    # чтобы голосовые сообщения тоже маршрутизировались в показ карточек, а не в LLM.
    if text and text.strip().startswith("/cards"):
        cards_filter = parse_cards_command_arguments(text.strip())
        if cards_filter is None:
            await tg_client.send_message(
                chat_id,
                "Формат:\n"
                "/cards — все активные (не в «Завершённые», не архив)\n"
                "/cards 7 — срок в ближайшие 7 календарных дней\n"
                "/cards сегодня | завтра | неделя\n"
                "/cards просрочка — дедлайн раньше сегодня\n\n"
                "Даты «сегодня» считаются в часовом поясе Europe/Moscow.",
            )
            return
        if not profile.trello_board_id:
            await tg_client.send_message(chat_id, BoardNotLinkedError().user_message)
            return
        try:
            lists_raw = await trello_client.list_lists(profile.trello_board_id)
            cards_raw = await trello_client.list_board_cards_summary(profile.trello_board_id)
            cards_raw = drop_cards_in_archived_lists(cards_raw, lists_raw)
            id_to_name = {str(x["id"]): str(x.get("name") or "") for x in lists_raw}
            messages = build_active_cards_messages(profile, cards_raw, id_to_name, cards_filter)
            for chunk in messages:
                await tg_client.send_message(chat_id, chunk)
        except TrelloAPIError as exc:
            await tg_client.send_message(chat_id, exc.user_message)
        await profile_repo.save(profile)
        return

    if text:
        nl_filter = parse_cards_nl_query(text.strip())
        if nl_filter is not None:
            if pending:
                await clarification_repo.clear(profile.telegram_user_id)
            if not profile.trello_board_id:
                await tg_client.send_message(chat_id, BoardNotLinkedError().user_message)
                return
            try:
                lists_raw = await trello_client.list_lists(profile.trello_board_id)
                cards_raw = await trello_client.list_board_cards_summary(profile.trello_board_id)
                cards_raw = drop_cards_in_archived_lists(cards_raw, lists_raw)
                id_to_name = {str(x["id"]): str(x.get("name") or "") for x in lists_raw}
                if nl_filter.kind == "today":
                    messages = build_active_cards_messages(profile, cards_raw, id_to_name, nl_filter)
                elif nl_filter.kind == "due_within" and (nl_filter.days or 0) <= 7:
                    messages = build_cards_by_days_window_messages(
                        profile,
                        cards_raw,
                        id_to_name,
                        nl_filter.days or 1,
                    )
                else:
                    messages = build_active_cards_messages(profile, cards_raw, id_to_name, nl_filter)
                for chunk in messages:
                    await tg_client.send_message(chat_id, chunk)
            except TrelloAPIError as exc:
                await tg_client.send_message(chat_id, exc.user_message)
            await profile_repo.save(profile)
            return

    if (
        not pending
        and text
        and text.strip()
        and not text.strip().startswith("/")
        and not profile.trello_board_id
    ):
        await tg_client.send_message(chat_id, BoardNotLinkedError().user_message)
        return

    try:
        intent_context = await intent_history_repo.recent_context(profile.telegram_user_id, limit=8)
        result = await orchestrator.process_text(profile, text, intent_context=intent_context)
        await profile_repo.save(profile)
        await tg_client.send_message(chat_id, result.text)
        try:
            action_type = result.action_type or quick_classify_intent(text) or "unknown"
            action_payload_json = json.dumps(result.action_payload, ensure_ascii=False) if result.action_payload else None
            await intent_history_repo.append(
                telegram_user_id=profile.telegram_user_id,
                user_text=text,
                action_type=action_type,
                response_text=result.text,
                action_payload_json=action_payload_json,
            )
        except Exception:  # noqa: BLE001
            logger.exception("intent history append failed for user_id=%s", profile.telegram_user_id)
    except BoardNotLinkedError as exc:
        await tg_client.send_message(chat_id, exc.user_message)
    except OrchestratorValidationError as exc:
        await tg_client.send_message(chat_id, exc.user_message)
    except TrelloAPIError as exc:
        await tg_client.send_message(chat_id, exc.user_message)
    except TelegramAPIError as exc:
        await tg_client.send_message(chat_id, exc.user_message)
    except OpenRouterLLMError as exc:
        await tg_client.send_message(chat_id, exc.user_message)
    except OpenRouterTranscriptionError as exc:
        await tg_client.send_message(chat_id, exc.user_message)
    except OpenAITranscriptionError as exc:
        await tg_client.send_message(chat_id, exc.user_message)
    except AppError as exc:
        await tg_client.send_message(chat_id, exc.user_message)
    except Exception as exc:
        await tg_client.send_message(chat_id, f"Нужно уточнение: {_fallback_message(exc)}")
