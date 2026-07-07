# 啟動 Flask（HTTP 模式、port 5000，與 Cloudflare Tunnel 搭配使用）
$port = 5000
$appProcess = Start-Process -FilePath "py" `
  -ArgumentList "-3.10", ".\app.py", "--http", "--port", $port `
  -WorkingDirectory $PSScriptRoot `
  -PassThru

Start-Sleep -Seconds 4

$cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"

Write-Host "Flask PID : $($appProcess.Id)"
Write-Host "Local URL : http://localhost:$port"
Write-Host "Starting Cloudflare Tunnel..."
Write-Host "Press Ctrl+C to stop the tunnel, then stop Python if needed."

& $cloudflared tunnel --url "http://localhost:$port"
