from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncConnection


def _user_profile_columns(sync_conn) -> set[str]:
    insp = inspect(sync_conn)
    if not insp.has_table("user_profiles"):
        return set()
    return {str(col["name"]) for col in insp.get_columns("user_profiles")}


async def ensure_user_profile_compat_columns(conn: AsyncConnection) -> None:
    """Add missing UserProfile columns for existing databases.

    `create_all` does not alter existing tables, so older installations may
    miss newly added columns.
    """
    cols = await conn.run_sync(_user_profile_columns)
    if not cols:
        return

    add_statements = {
        "timezone": "ALTER TABLE user_profiles ADD COLUMN timezone VARCHAR(64) DEFAULT 'Europe/Moscow'",
        "routine_cron": "ALTER TABLE user_profiles ADD COLUMN routine_cron VARCHAR(64) DEFAULT '0 8 * * *'",
        "motivator_cron_windows": "ALTER TABLE user_profiles ADD COLUMN motivator_cron_windows VARCHAR(128) DEFAULT '0 11,16,20 * * *'",
        "analytics_cron": "ALTER TABLE user_profiles ADD COLUMN analytics_cron VARCHAR(64) DEFAULT '30 21 * * *'",
    }

    for col_name, stmt in add_statements.items():
        if col_name in cols:
            continue
        await conn.exec_driver_sql(stmt)

    # Backfill NULLs in case some DBs do not populate defaults for old rows.
    await conn.exec_driver_sql(
        "UPDATE user_profiles SET timezone='Europe/Moscow' WHERE timezone IS NULL OR timezone=''"
    )
    await conn.exec_driver_sql(
        "UPDATE user_profiles SET routine_cron='0 8 * * *' WHERE routine_cron IS NULL OR routine_cron=''"
    )
    await conn.exec_driver_sql(
        "UPDATE user_profiles SET motivator_cron_windows='0 11,16,20 * * *' "
        "WHERE motivator_cron_windows IS NULL OR motivator_cron_windows=''"
    )
    await conn.exec_driver_sql(
        "UPDATE user_profiles SET analytics_cron='30 21 * * *' WHERE analytics_cron IS NULL OR analytics_cron=''"
    )

