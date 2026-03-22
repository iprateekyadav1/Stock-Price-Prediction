#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

TICKER="${TICKER:-RELIANCE.NS}"
PERIOD="${PERIOD:-5y}"
EPOCHS="${EPOCHS:-25}"
TOP_N="${TOP_N:-10}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8001}"
FRONTEND_DIR="${ROOT_DIR}/frontend"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Python was not found. Install Python 3 first."
  exit 1
fi

create_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[setup] Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "This script expects a POSIX virtualenv layout."
    echo "Run it from Git Bash/WSL/macOS/Linux, or adapt it for PowerShell on Windows."
    exit 1
  fi

  PYTHON_BIN="${VENV_DIR}/bin/python"
}

install_deps() {
  create_venv
  echo "[setup] Installing dependencies"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"
}

ensure_ready() {
  create_venv

  if ! "${PYTHON_BIN}" -c "import torch, pandas, sklearn, yfinance, fastapi, uvicorn" >/dev/null 2>&1; then
    install_deps
  fi
}

ensure_frontend_ready() {
  if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
    echo "[frontend] Installing npm packages"
    (cd "${FRONTEND_DIR}" && npm install)
  fi
}

build_frontend() {
  ensure_frontend_ready
  (cd "${FRONTEND_DIR}" && npm run build)
}

run_train() {
  ensure_ready
  "${PYTHON_BIN}" "${ROOT_DIR}/bot.py" train "${TICKER}" --period "${PERIOD}" --epochs "${EPOCHS}"
}

run_pretrain() {
  ensure_ready
  "${PYTHON_BIN}" "${ROOT_DIR}/bot.py" pretrain --period "${PERIOD}" --epochs "${EPOCHS}"
}

run_scan() {
  ensure_ready
  "${PYTHON_BIN}" "${ROOT_DIR}/bot.py" scan --top "${TOP_N}"
}

run_analyze() {
  ensure_ready
  "${PYTHON_BIN}" "${ROOT_DIR}/bot.py" analyze "${TICKER}" --period "${PERIOD}"
}

run_advise() {
  ensure_ready
  "${PYTHON_BIN}" "${ROOT_DIR}/bot.py" advise "${TICKER}"
}

run_full() {
  ensure_ready

  if [[ ! -f "${ROOT_DIR}/best_lstm_model.pth" || ! -f "${ROOT_DIR}/scalers.pkl" ]]; then
    echo "[full] Model artifacts missing, training first"
    run_train
  else
    echo "[full] Reusing existing model artifacts"
  fi

  echo "[full] Running market scan"
  run_scan

  echo "[full] Running backtest analysis for ${TICKER}"
  run_analyze

  echo "[full] Launching advisor for ${TICKER}"
  run_advise
}

run_api() {
  ensure_ready
  build_frontend
  "${PYTHON_BIN}" -m uvicorn api_server:app --host "${API_HOST}" --port "${API_PORT}"
}

run_frontend() {
  echo "[frontend] Building frontend bundle for FastAPI to serve"
  build_frontend
  echo "[frontend] Build ready at frontend/dist"
}

run_dashboard() {
  ensure_ready
  build_frontend
  echo "[dashboard] Serving dashboard and API on http://${API_HOST}:${API_PORT}"
  "${PYTHON_BIN}" -m uvicorn api_server:app --host "${API_HOST}" --port "${API_PORT}"
}

print_help() {
  cat <<EOF
Usage: ./run_project.sh <command>

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
  help     Show this help text

Environment overrides:
  TICKER=RELIANCE.NS
  PERIOD=5y
  EPOCHS=25
  TOP_N=10
  API_HOST=127.0.0.1
  API_PORT=8001

Examples:
  ./run_project.sh setup
  ./run_project.sh pretrain
  ./run_project.sh full
  ./run_project.sh api
  ./run_project.sh frontend
  ./run_project.sh dashboard
  TICKER=TCS.NS EPOCHS=50 ./run_project.sh train
  TICKER=INFY.NS ./run_project.sh analyze
EOF
}

case "${1:-help}" in
  setup)
    install_deps
    ;;
  train)
    run_train
    ;;
  pretrain)
    run_pretrain
    ;;
  scan)
    run_scan
    ;;
  analyze)
    run_analyze
    ;;
  advise)
    run_advise
    ;;
  full)
    run_full
    ;;
  api)
    run_api
    ;;
  frontend)
    run_frontend
    ;;
  dashboard)
    run_dashboard
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    echo "Unknown command: $1"
    echo
    print_help
    exit 1
    ;;
esac
