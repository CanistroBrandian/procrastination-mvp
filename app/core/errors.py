"""Ошибки по подсистемам — в Telegram уходит понятный текст, не сырой traceback."""


class AppError(Exception):
    """Базовая ошибка приложения с сообщением для пользователя."""

    def __init__(self, user_message: str, detail: str | None = None):
        self.user_message = user_message
        self.detail = detail
        super().__init__(detail or user_message)


class BoardNotLinkedError(AppError):
    """Нет привязанной Trello-доски у пользователя."""

    def __init__(self) -> None:
        super().__init__(
            user_message=(
                "Доска Trello не привязана. Сначала: /boards — скопируйте id доски — затем /link <board_id>."
            ),
        )


class OrchestratorValidationError(AppError):
    """Некорректные данные для действия (без HTTP)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TrelloAPIError(AppError):
    """Ошибка HTTP/сети при обращении к api.trello.com."""

    def __init__(self, message: str, *, status_code: int | None = None, url: str | None = None):
        self.status_code = status_code
        self.url = url
        super().__init__(message, detail=message)


class TelegramAPIError(AppError):
    """Ошибка при вызове api.telegram.org (кроме бизнес-логики)."""

    def __init__(self, user_message: str, *, status_code: int | None = None, detail: str | None = None):
        self.status_code = status_code
        super().__init__(user_message, detail=detail)


class OpenRouterLLMError(AppError):
    """Ошибка чат-модели через OpenRouter / OpenAI SDK (не транскрипция)."""


class OpenRouterTranscriptionError(AppError):
    """Ошибка Whisper через OpenRouter REST (audio/transcriptions)."""


class OpenAITranscriptionError(AppError):
    """Ошибка Whisper через официальный OpenAI Audio API."""
