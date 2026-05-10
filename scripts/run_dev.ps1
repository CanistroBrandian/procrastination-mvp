# Локальный запуск API (http://127.0.0.1:8000). В .env задайте реальные ключи и APP_BASE_URL для webhook.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
