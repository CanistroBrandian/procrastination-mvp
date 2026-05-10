# Подготовка окружения: .env, зависимости, тесты.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path (Join-Path $Root ".env"))) {
    Copy-Item (Join-Path $Root ".env.example") (Join-Path $Root ".env")
    Write-Host "Создан .env из .env.example — откройте и вставьте ТОЛЬКО НОВЫЕ ключи (после ротации)."
}

python -m pip install -r requirements.txt
python -m pytest -q
Write-Host "Готово."
