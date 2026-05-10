from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.llm_client import build_async_openai_client
from app.db.session import get_db_session
from app.services.telegram_pipeline import process_telegram_update

router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.post("/webhook")
async def telegram_webhook(
    update: dict,
    db: AsyncSession = Depends(get_db_session),
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict:
    cfg = get_settings()
    openai_client = build_async_openai_client(cfg)
    if cfg.telegram_webhook_secret and x_telegram_bot_api_secret_token != cfg.telegram_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid secret token")

    await process_telegram_update(db, update, cfg, openai_client)
    return {"ok": True}
