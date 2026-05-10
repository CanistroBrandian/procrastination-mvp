# Quick tunnel: публичный HTTPS -> localhost:8000. Скопируйте https://....trycloudflare.com в APP_BASE_URL и перезапустите API.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$cf = Join-Path $Root "tools\cloudflared.exe"
if (-not (Test-Path $cf)) {
    Write-Host "cloudflared не найден. Скачайте: https://github.com/cloudflare/cloudflared/releases"
    exit 1
}
Write-Host "Туннель к http://127.0.0.1:8000 (сначала запустите run_dev.ps1 в другом окне)"
# http2 часто проходит там, где QUIC режется VPN/файрволом.
& $cf tunnel --protocol http2 --url http://127.0.0.1:8000
