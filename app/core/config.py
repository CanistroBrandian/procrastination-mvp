from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    database_url: str = "sqlite+aiosqlite:///./app.db"
    app_base_url: str = "http://localhost:8000"
    telegram_bot_token: str = ""
    # polling — long polling getUpdates (без HTTPS). webhook — только если есть публичный APP_BASE_URL.
    telegram_mode: str = "polling"
    telegram_webhook_secret: str = ""
    openai_api_key: str = ""
    # OpenRouter: один ключ для Whisper (audio/transcriptions) и для chat (OpenAI-compatible API).
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str = ""
    openrouter_app_title: str = ""
    whisper_model: str = "openai/whisper-large-v3"
    chat_model: str = "openai/gpt-4o-mini"
    trello_api_key: str = ""
    trello_api_token: str = ""
    default_trello_board_id: str = ""
    reminder_cron: str = "*/30 * * * *"


def get_settings() -> Settings:
    """Без кэша: изменения в .env подхватываются после перезапуска процесса."""
    return Settings()
