$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RootDir ".venv\Scripts\python.exe"
$ApiHost = if ($env:API_HOST) { $env:API_HOST } else { "127.0.0.1" }
$ApiPort = if ($env:API_PORT) { $env:API_PORT } else { "8001" }

Set-Location $RootDir

if (-not (Test-Path $Python)) {
    throw "Python virtual environment not found at $Python"
}

& $Python -m uvicorn api_server:app --host $ApiHost --port $ApiPort
