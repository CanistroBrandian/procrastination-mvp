from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from openai import AsyncOpenAI

# Включаем INFO-логи для нашего пакета. Это позволяет видеть работу
# двухшагового агента (intent_router → intent_extractor) в консоли uvicorn.
# basicConfig обязателен — без него Python шлёт логи через lastResort на WARNING.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
logging.getLogger("app").setLevel(logging.INFO)

from app.api.webhook import router as webhook_router
from app.core.config import get_settings
from app.core.llm_client import build_async_openai_client
from app.db.models import Base
from app.db.repositories import IntentHistoryRepository, RoutineTemplateRepository, TaskEventRepository, UserProfileRepository
from app.db.session import SessionLocal, engine
from app.integrations.telegram import TelegramClient
from app.integrations.trello import TrelloClient
from app.workers.analytics_report import run_daily_analytics_report
from app.workers.motivator import run_overdue_motivator
from app.workers.reminders import run_reminders
from app.workers.routines import run_routine_generation
from app.workers.telegram_polling import telegram_polling_loop

scheduler = AsyncIOScheduler()


async def reminder_job() -> None:
    cfg = get_settings()
    tg = TelegramClient(cfg.telegram_bot_token)
    trello = TrelloClient(cfg.trello_api_key, cfg.trello_api_token)
    async with SessionLocal() as session:
        repo = UserProfileRepository(session)
        await run_reminders(repo, tg, trello)


async def routine_generation_job() -> None:
    cfg = get_settings()
    trello = TrelloClient(cfg.trello_api_key, cfg.trello_api_token)
    async with SessionLocal() as session:
        profile_repo = UserProfileRepository(session)
        routines_repo = RoutineTemplateRepository(session)
        events_repo = TaskEventRepository(session)
        await run_routine_generation(profile_repo, routines_repo, events_repo, trello)


async def analytics_report_job() -> None:
    cfg = get_settings()
    tg = TelegramClient(cfg.telegram_bot_token)
    async with SessionLocal() as session:
        profile_repo = UserProfileRepository(session)
        events_repo = TaskEventRepository(session)
        await run_daily_analytics_report(profile_repo, events_repo, tg)


async def overdue_motivator_job() -> None:
    cfg = get_settings()
    tg = TelegramClient(cfg.telegram_bot_token)
    trello = TrelloClient(cfg.trello_api_key, cfg.trello_api_token)
    async with SessionLocal() as session:
        profile_repo = UserProfileRepository(session)
        events_repo = TaskEventRepository(session)
        history_repo = IntentHistoryRepository(session)
        await run_overdue_motivator(profile_repo, events_repo, history_repo, tg, trello)


def _cron_kwargs(expr: str) -> dict[str, str]:
    minute, hour, day, month, day_of_week = expr.split()
    return {
        "minute": minute,
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    poll_stop = asyncio.Event()
    poll_task: asyncio.Task | None = None
    cfg = get_settings()
    mode = cfg.telegram_mode.strip().lower()
    openai_client = build_async_openai_client(cfg)

    if mode == "polling" and not cfg.telegram_bot_token:
        import logging

        logging.getLogger("uvicorn.error").warning(
            "TELEGRAM_BOT_TOKEN пуст — бот не будет отвечать. Заполните .env и перезапустите сервер."
        )

    if mode == "polling" and cfg.telegram_bot_token:
        poll_task = asyncio.create_task(telegram_polling_loop(cfg, poll_stop, openai_client))
    elif mode == "webhook" and cfg.telegram_bot_token and cfg.app_base_url:
        tg = TelegramClient(cfg.telegram_bot_token)
        try:
            await tg.set_webhook(
                f"{cfg.app_base_url.rstrip('/')}/telegram/webhook",
                cfg.telegram_webhook_secret,
            )
        except Exception as exc:
            import logging

            logging.getLogger("uvicorn.error").warning("Telegram setWebhook failed: %s", exc)

    if not scheduler.running:
        scheduler.add_job(
            reminder_job,
            "cron",
            id="reminders",
            replace_existing=True,
            **_cron_kwargs(cfg.reminder_cron),
        )
        scheduler.add_job(
            routine_generation_job,
            "cron",
            id="routine_generation",
            replace_existing=True,
            **_cron_kwargs(cfg.routine_worker_cron),
        )
        scheduler.add_job(
            overdue_motivator_job,
            "cron",
            id="overdue_motivator",
            replace_existing=True,
            **_cron_kwargs(cfg.motivator_worker_cron),
        )
        scheduler.add_job(
            analytics_report_job,
            "cron",
            id="daily_analytics_report",
            replace_existing=True,
            **_cron_kwargs(cfg.analytics_worker_cron),
        )
        scheduler.start()
    yield
    poll_stop.set()
    if poll_task:
        try:
            await asyncio.wait_for(poll_task, timeout=25)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            poll_task.cancel()
    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(title="Procrastination Assistant MVP", lifespan=lifespan)
app.include_router(webhook_router)


@app.get("/")
async def service_info() -> dict:
    """Куда стучится Telegram и кто может вызывать HTTP (для webhook)."""
    cfg = get_settings()
    mode = cfg.telegram_mode.strip().lower()
    webhook_path = "/telegram/webhook"
    base = cfg.app_base_url.rstrip("/")
    return {
        "service": "procrastination-assistant-mvp",
        "telegram": {
            "mode": mode,
            "polling": {
                "note": "Бот сам запрашивает обновления у api.telegram.org (long polling). "
                "Входящие HTTP от Telegram не нужны — только исходящий интернет.",
            },
            "webhook": {
                "note": "Telegram шлёт POST только если вы выставили webhook.",
                "post_url": f"{base}{webhook_path}" if base else webhook_path,
                "secret_header": "X-Telegram-Bot-Api-Secret-Token",
                "secret_must_match": "TELEGRAM_WEBHOOK_SECRET в .env (если задан — без заголовка запрос отклонится)",
            },
        },
        "security": {
            "who_should_call_this_server": [
                "При polling — только вы локально; Telegram сам отдаёт сообщения через getUpdates.",
                "При webhook — только серверы Telegram на ваш HTTPS URL; защита заголовком секрета.",
                "Публично вывешивайте только HTTPS URL за reverse-proxy; не открывайте bare HTTP в интернет без TLS.",
            ],
        },
    }


@app.get("/health")
async def health() -> dict:
    cfg = get_settings()
    mode = cfg.telegram_mode.strip().lower()
    token_ok = bool(cfg.telegram_bot_token)
    return {
        "ok": True,
        "env": cfg.app_env,
        "model": AsyncOpenAI.__name__,
        "telegram_mode": mode,
        "telegram_token_configured": token_ok,
        "polling_should_run": mode == "polling" and token_ok,
    }
