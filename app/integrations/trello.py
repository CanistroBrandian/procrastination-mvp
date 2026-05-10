from __future__ import annotations

from typing import Any

import httpx

from app.core.errors import TrelloAPIError


class TrelloClient:
    def __init__(self, api_key: str, api_token: str):
        self.api_key = (api_key or "").strip()
        self.api_token = (api_token or "").strip()
        self.base_url = "https://api.trello.com/1"

    def _auth_params(self) -> dict[str, str]:
        return {"key": self.api_key, "token": self.api_token}

    def _ensure_credentials(self) -> None:
        """Пустые key/token дают у Trello часто HTTP 400 — режем заранее с понятным текстом."""
        if not self.api_key or not self.api_token:
            raise TrelloAPIError(
                "Trello: в .env не заданы TRELLO_API_KEY и/или TRELLO_API_TOKEN. "
                "Возьмите ключ на https://trello.com/app-key, нажмите Token — вставьте токен в TRELLO_API_TOKEN.",
            )

    def _trello_body_snippet(self, response: httpx.Response, limit: int = 280) -> str:
        try:
            t = response.text.strip()
            if len(t) > limit:
                return t[:limit] + "…"
            return t
        except Exception:
            return ""

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        empty_body_ok: bool = False,
    ) -> Any:
        self._ensure_credentials()
        query = {**self._auth_params(), **(params or {})}
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.request(method, url, params=query)
                response.raise_for_status()
                if empty_body_ok and not (response.text or "").strip():
                    return {}
                return response.json()
        except httpx.HTTPStatusError as exc:
            body = self._trello_body_snippet(exc.response)
            body_l = body.lower()
            hint_map = {
                400: (
                    "Обычно: пустой/чужой token, токен выдан не для этого API key, опечатка, или неверный id. "
                    "Перевыпустите token на странице ключа: https://trello.com/app-key"
                ),
                401: (
                    "Неверная пара key+token. Важно: в TRELLO_API_TOKEN нужен не «Secret» со страницы приложения, "
                    "а пользовательский Token: на https://trello.com/app-key нажмите «Token» → авторизуйтесь → "
                    "скопируйте длинную строку в TRELLO_API_TOKEN. В TRELLO_API_KEY — только поле API key (32 символа). "
                    "Secret используется для OAuth-потоков, не подставляйте его вместо token в REST."
                ),
                403: "Доступ запрещён — проверьте права токена и членство на доске.",
                404: "Ресурс не найден — проверьте id доски/карточки.",
            }
            hint = hint_map.get(
                exc.response.status_code,
                "См. ответ Trello ниже; проверьте TRELLO_API_KEY, TRELLO_API_TOKEN и интернет.",
            )
            if exc.response.status_code == 401 and "invalid key" in body_l:
                hint += (
                    " Ответ «invalid key» значит неверный TRELLO_API_KEY: скопируйте ключ заново с "
                    "https://trello.com/app-key (без пробелов и кавычек)."
                )
            detail = f"Trello: HTTP {exc.response.status_code}. {hint}"
            if body:
                detail += f" Детали API: {body}"
            raise TrelloAPIError(
                detail,
                status_code=exc.response.status_code,
                url=str(exc.request.url),
            ) from exc
        except httpx.TimeoutException as exc:
            raise TrelloAPIError(
                "Trello: таймаут сети. Проверьте интернет и доступность api.trello.com.",
            ) from exc
        except httpx.RequestError as exc:
            raise TrelloAPIError(
                "Trello: нет соединения с api.trello.com (сеть, VPN, файрвол). "
                "Проверьте интернет и переменные TRELLO_* в .env.",
            ) from exc

    async def list_boards(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/members/me/boards")

    async def list_lists(self, board_id: str) -> list[dict[str, Any]]:
        return await self._request("GET", f"/boards/{board_id}/lists")

    async def list_board_cards_with_checklists(self, board_id: str) -> list[dict[str, Any]]:
        """Карточки доски вместе с чеклистами и пунктами (для поиска по тексту)."""
        return await self._request(
            "GET",
            f"/boards/{board_id}/cards",
            {
                # visible по умолчанию может не отдавать часть карточек; берём all и режем closed сами.
                "filter": "all",
                "checklists": "all",
                "checklist_fields": "all",
                "fields": "name,idList,closed,shortUrl",
            },
        )

    async def list_board_cards_summary(self, board_id: str) -> list[dict[str, Any]]:
        """Все карточки доски с полями для списка активных задач (без чеклистов — быстрее)."""
        return await self._request(
            "GET",
            f"/boards/{board_id}/cards",
            {
                "filter": "all",
                "fields": "name,idList,closed,shortUrl,due,dueComplete",
            },
        )

    async def get_card_with_checklists(self, card_id: str) -> dict[str, Any]:
        """Одна карточка с полными чеклистами и пунктами (для массового завершения пунктов)."""
        return await self._request(
            "GET",
            f"/cards/{card_id}",
            {
                "checklists": "all",
                "checklist_fields": "all",
                "fields": "name,idList,closed,shortUrl",
            },
        )

    async def list_card_checklists(self, card_id: str) -> list[dict[str, Any]]:
        """Чеклисты карточки с пунктами — надёжнее, чем вложение в GET /cards/{id} (Trello не всегда отдаёт checkItems)."""
        return await self._request(
            "GET",
            f"/cards/{card_id}/checklists",
            {
                "checkItems": "all",
                "checkItem_fields": "all",
            },
        )

    async def create_card(
        self,
        list_id: str,
        name: str,
        desc: str = "",
        *,
        due: str | None = None,
        start: str | None = None,
        due_complete: bool | None = None,
        position: str | float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"idList": list_id, "name": name, "desc": desc}
        if due is not None:
            params["due"] = due
        if start is not None:
            params["start"] = start
        if due_complete is not None:
            params["dueComplete"] = due_complete
        if position is not None:
            params["pos"] = position
        return await self._request("POST", "/cards", params)

    async def update_card(
        self,
        card_id: str,
        name: str | None = None,
        desc: str | None = None,
        *,
        due: str | None = None,
        start: str | None = None,
        due_complete: bool | None = None,
        closed: bool | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if name is not None:
            params["name"] = name
        if desc is not None:
            params["desc"] = desc
        if due is not None:
            params["due"] = due
        if start is not None:
            params["start"] = start
        if due_complete is not None:
            params["dueComplete"] = due_complete
        if closed is not None:
            params["closed"] = closed
        return await self._request("PUT", f"/cards/{card_id}", params)

    async def delete_card(self, card_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/cards/{card_id}", empty_body_ok=True)

    async def move_card(self, card_id: str, list_id: str) -> dict[str, Any]:
        return await self._request("PUT", f"/cards/{card_id}", {"idList": list_id})

    async def add_checklist(self, card_id: str, name: str) -> dict[str, Any]:
        return await self._request("POST", "/checklists", {"idCard": card_id, "name": name})

    async def add_check_item(self, checklist_id: str, item_name: str) -> dict[str, Any]:
        return await self._request("POST", f"/checklists/{checklist_id}/checkItems", {"name": item_name})

    async def attach_file_by_url(self, card_id: str, file_url: str, name: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"url": file_url}
        if name:
            params["name"] = name
        return await self._request("POST", f"/cards/{card_id}/attachments", params)

    async def add_card_comment(self, card_id: str, text: str) -> dict[str, Any]:
        return await self._request("POST", f"/cards/{card_id}/actions/comments", {"text": text})

    async def add_label_to_card(self, card_id: str, label_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/cards/{card_id}/idLabels", {"value": label_id})

    async def remove_label_from_card(self, card_id: str, label_id: str) -> dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/cards/{card_id}/idLabels/{label_id}",
            empty_body_ok=True,
        )

    async def add_member_to_card(self, card_id: str, member_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/cards/{card_id}/idMembers", {"value": member_id})

    async def remove_member_from_card(self, card_id: str, member_id: str) -> dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/cards/{card_id}/idMembers/{member_id}",
            empty_body_ok=True,
        )

    async def update_check_item_on_card(
        self,
        card_id: str,
        checklist_id: str,
        check_item_id: str,
        *,
        complete: bool | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if complete is not None:
            params["state"] = "complete" if complete else "incomplete"
        if name is not None:
            params["name"] = name
        return await self._request(
            "PUT",
            f"/cards/{card_id}/checklist/{checklist_id}/checkItem/{check_item_id}",
            params,
        )
