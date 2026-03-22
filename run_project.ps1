param(
    [ValidateSet("setup", "train", "pretrain", "scan", "analyze", "advise", "full", "api", "frontend", "dashboard", "open", "help")]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPythonw = Join-Path $VenvDir "Scripts\pythonw.exe"

$Ticker = if ($env:TICKER) { $env:TICKER } else { "RELIANCE.NS" }
$Period = if ($env:PERIOD) { $env:PERIOD } else { "5y" }
$Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "25" }
$TopN = if ($env:TOP_N) { $env:TOP_N } else { "10" }
$ApiHost = if ($env:API_HOST) { $env:API_HOST } else { "127.0.0.1" }
$ApiPort = if ($env:API_PORT) { $env:API_PORT } else { "8001" }
$FrontendDir = Join-Path $RootDir "frontend"

function Get-PythonCommand {
    if (Test-Path $VenvPython) {
        return $VenvPython
    }

    $python = Get-Command python, py -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $python) {
        throw "Python was not found. Install Python 3 first."
    }

    return $python.Source
}

function Create-Venv {
    if (-not (Test-Path $VenvPython)) {
        $python = Get-PythonCommand
        Write-Host "[setup] Creating virtual environment at $VenvDir"
        & $python -m venv $VenvDir
    }
}

function Install-Deps {
    Create-Venv
    Write-Host "[setup] Installing dependencies"
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $RootDir "requirements.txt")
}

function Ensure-Ready {
    Create-Venv

    & $VenvPython -c "import torch, pandas, sklearn, yfinance, fastapi, uvicorn" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Install-Deps
    }
}

function Ensure-FrontendReady {
    if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
        Write-Host "[frontend] Installing npm packages"
        Push-Location $FrontendDir
        try {
            npm.cmd install
        } finally {
            Pop-Location
        }
    }
}

function Build-Frontend {
    Ensure-FrontendReady
    Push-Location $FrontendDir
    try {
        npm.cmd run build
    } finally {
        Pop-Location
    }
}

function Run-Train {
    Ensure-Ready
    & $VenvPython (Join-Path $RootDir "bot.py") train $Ticker --period $Period --epochs $Epochs
}

function Run-Pretrain {
    Ensure-Ready
    & $VenvPython (Join-Path $RootDir "bot.py") pretrain --period $Period --epochs $Epochs
}

function Run-Scan {
    Ensure-Ready
    & $VenvPython (Join-Path $RootDir "bot.py") scan --top $TopN
}

function Run-Analyze {
    Ensure-Ready
    & $VenvPython (Join-Path $RootDir "bot.py") analyze $Ticker --period $Period
}

function Run-Advise {
    Ensure-Ready
    & $VenvPython (Join-Path $RootDir "bot.py") advise $Ticker
}

function Run-Full {
    Ensure-Ready

    $modelPath = Join-Path $RootDir "best_lstm_model.pth"
    $scalerPath = Join-Path $RootDir "scalers.pkl"

    if (-not (Test-Path $modelPath) -or -not (Test-Path $scalerPath)) {
        Write-Host "[full] Model artifacts missing, training first"
        Run-Train
    } else {
        Write-Host "[full] Reusing existing model artifacts"
    }

    Write-Host "[full] Running market scan"
    Run-Scan

    Write-Host "[full] Running backtest analysis for $Ticker"
    Run-Analyze

    Write-Host "[full] Launching advisor for $Ticker"
    Run-Advise
}

function Run-Api {
    Ensure-Ready
    Build-Frontend
    & $VenvPython -m uvicorn api_server:app --host $ApiHost --port $ApiPort
}

function Run-Frontend {
    Write-Host "[frontend] Building frontend bundle for FastAPI to serve"
    Build-Frontend
    Write-Host "[frontend] Build ready at frontend/dist"
}

function Run-Dashboard {
    Ensure-Ready
    Build-Frontend

    Write-Host "[dashboard] Serving dashboard and API on http://$ApiHost`:$ApiPort"
    & $VenvPython -m uvicorn api_server:app --host $ApiHost --port $ApiPort
}

function Open-DashboardWindow {
    Ensure-Ready
    Build-Frontend

    Write-Host "[dashboard] Starting dashboard in a minimized PowerShell window"
    Start-Process powershell -WindowStyle Minimized -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $RootDir "start_dashboard_background.ps1")
    ) | Out-Null
}

function Show-Help {
    @"
Usage: .\run_project.ps1 <command>

Commands:
  setup    Create .venv and install dependencies
  train    Train the LSTM model for TICKER
  pretrain Train the starter model pack
  scan     Run the stock screener
  analyze  Run the backtest for TICKER
  advise   Run the interactive advisor for TICKER
  full     Run scan + analyze + advise, training first only if artifacts are missing
  api      Build frontend and start the FastAPI server
  frontend Build the React dashboard bundle
  dashboard Build frontend and serve dashboard + API together
  open     Start the dashboard server in the background on Windows
  help     Show this help text

Environment overrides:
  TICKER=RELIANCE.NS
  PERIOD=5y
  EPOCHS=25
  TOP_N=10
  API_HOST=127.0.0.1
  API_PORT=8001

Examples:
  .\run_project.ps1 setup
  .\run_project.ps1 pretrain
  .\run_project.ps1 full
  .\run_project.ps1 api
  .\run_project.ps1 frontend
  .\run_project.ps1 dashboard
  .\run_project.ps1 open
  `$env:TICKER='TCS.NS'; `$env:EPOCHS='50'; .\run_project.ps1 train
  `$env:TICKER='INFY.NS'; .\run_project.ps1 analyze
"@
}

switch ($Command) {
    "setup" { Install-Deps }
    "train" { Run-Train }
    "pretrain" { Run-Pretrain }
    "scan" { Run-Scan }
    "analyze" { Run-Analyze }
    "advise" { Run-Advise }
    "full" { Run-Full }
    "api" { Run-Api }
    "frontend" { Run-Frontend }
    "dashboard" { Run-Dashboard }
    "open" { Open-DashboardWindow }
    default { Show-Help }
}
