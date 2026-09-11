@echo off
TITLE Antigravity Trading System - One Click Start
SETLOCAL

:: Navigate to the directory where this script is located
cd /d "%~dp0"

echo [1/4] Activating Virtual Environment...
IF EXIST ".venv\Scripts\activate" (
    CALL .venv\Scripts\activate
) ELSE (
    echo [WARNING] .venv not found. Running with system python...
)

echo [2/4] Starting SMC Dashboard (View Candles)...
start "SMC Dashboard" cmd /k "python tools/view_candles.py"

echo [3/4] Starting Live Price Scanner...
start "Live Scanner" cmd /k "python live_scanner.py"

echo [4/4] Starting Challenge Trader Engine...
start "Challenge Trader" cmd /k "python challenge_trader.py"

echo.
echo ======================================================
echo   SUCCESS: All 3 systems are starting in new windows.
echo   - Close the windows to stop the servers.
echo ======================================================
pause
