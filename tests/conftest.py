"""Чтобы тесты не запускали Telegram long polling и не зависали."""
import os

# До импорта app.main: выключаем polling и токен — lifespan не создаёт цикл getUpdates.
os.environ.setdefault("TELEGRAM_MODE", "webhook")
os.environ["TELEGRAM_BOT_TOKEN"] = ""
