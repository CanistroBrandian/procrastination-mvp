from __future__ import annotations

from datetime import datetime

from app.db.models import RoutineTemplate
from app.services.routines import RoutineService


def _template(**kwargs):
    return RoutineTemplate(
        telegram_user_id=1,
        name=kwargs.get("name", "English"),
        category_key=kwargs.get("category_key", "study"),
        duration_min=kwargs.get("duration_min", 60),
        checklist_template_json=kwargs.get("checklist_template_json", '["lesson","review"]'),
        schedule_cron=kwargs.get("schedule_cron", "0 8 * * *"),
        is_active=kwargs.get("is_active", 1),
        last_generated_date=kwargs.get("last_generated_date"),
    )


def test_should_run_template_is_idempotent_per_day():
    now_local = datetime(2026, 5, 13, 8, 0, 0)
    template = _template(last_generated_date=None)
    assert RoutineService.should_run_template(template, now_local) is True

    template.last_generated_date = "2026-05-13"
    assert RoutineService.should_run_template(template, now_local) is False


def test_parse_checklist_template_json():
    template = _template(checklist_template_json='["grammar","speaking"," "]')
    items = RoutineService.parse_checklist(template)
    assert items == ["grammar", "speaking"]

