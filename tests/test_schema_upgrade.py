from __future__ import annotations

import asyncio

from app.db.schema_upgrade import ensure_user_profile_compat_columns


class FakeConn:
    def __init__(self, existing_cols: set[str]):
        self.existing_cols = existing_cols
        self.sql: list[str] = []

    async def run_sync(self, fn):
        class _SyncConn:  # pragma: no cover - helper stub
            pass

        # We bypass inspector internals: test behavior by monkeypatching callback result.
        return self.existing_cols

    async def exec_driver_sql(self, stmt: str):
        self.sql.append(stmt)


def _run(coro):
    return asyncio.run(coro)


def test_schema_upgrade_adds_missing_user_profile_columns():
    conn = FakeConn(existing_cols={"id", "telegram_user_id"})
    _run(ensure_user_profile_compat_columns(conn))  # type: ignore[arg-type]

    joined = "\n".join(conn.sql)
    assert "ADD COLUMN timezone" in joined
    assert "ADD COLUMN routine_cron" in joined
    assert "ADD COLUMN motivator_cron_windows" in joined
    assert "ADD COLUMN analytics_cron" in joined
    assert "UPDATE user_profiles SET timezone='Europe/Moscow'" in joined

