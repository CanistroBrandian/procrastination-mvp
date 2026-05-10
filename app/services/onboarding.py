from __future__ import annotations

from app.db.models import UserProfile
from app.integrations.trello import TrelloClient


def _match_inbox(name: str) -> bool:
    n = (name or "").lower().strip()
    if not n:
        return False
    if n in {"to do", "to-do", "входящие"}:
        return True
    markers = (
        "inbox",
        "todo",
        "to do",
        "backlog",
        "бэклог",
        "беклог",
        "инбокс",
        "надо сделать",
        "к выполнению",
        "incoming",
    )
    return any(m in n for m in markers)


def _match_doing(name: str) -> bool:
    n = (name or "").lower().strip()
    if not n:
        return False
    markers = (
        "doing",
        "in progress",
        "progress",
        "wip",
        "в работе",
        "development",
        "develop",
        "doing now",
    )
    return any(m in n for m in markers)


def _match_done(name: str) -> bool:
    n = (name or "").lower().strip()
    if not n:
        return False
    # не путаем с «incomplete» / «не завершено»
    if "incomplete" in n or "не заверш" in n:
        return False
    # Колонка inbox/backlog не может быть «завершённой»: иначе совпадения вроде
    # «Product Release Backlog» дают и backlog, и маркер release → done_id = id бэклога,
    # и /cards скрывает весь столбец.
    if _match_inbox(name):
        return False
    markers = (
        "done",
        "completed",
        "complete",
        "released",
        "release",
        "заверш",
        "выполн",
        "готово",
        "сделано",
        "закрыт",
        "архив задач",
    )
    return any(m in n for m in markers)


class OnboardingService:
    def __init__(self, trello_client: TrelloClient):
        self.trello_client = trello_client

    async def link_default_board(self, profile: UserProfile, board_id: str) -> None:
        lists = await self.trello_client.list_lists(board_id)
        mapped = {"inbox": None, "doing": None, "done": None}
        for item in lists:
            if item.get("closed"):
                continue
            name = str(item.get("name") or "")
            if _match_inbox(name):
                mapped["inbox"] = item["id"]
            if _match_doing(name):
                mapped["doing"] = item["id"]
            if _match_done(name):
                mapped["done"] = item["id"]
        profile.trello_board_id = board_id
        open_lists = [x for x in lists if not x.get("closed")]
        profile.trello_inbox_list_id = mapped["inbox"] or (open_lists[0]["id"] if open_lists else None)
        profile.trello_doing_list_id = mapped["doing"] or profile.trello_inbox_list_id
        # Не подставляем doing/inbox вместо done: иначе весь столбец ошибочно считается «завершённым»
        # (типичный случай — русские названия без подстроки "done").
        profile.trello_done_list_id = mapped["done"]
