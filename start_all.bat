@echo off
TITLE Trading System Fleet Launcher
COLOR 0A

echo ==============================================================
echo           TRADING SUITE & DUAL CHALLENGE LAUNCHER
echo ==============================================================
echo.
echo Launching:
echo   1. Challenge Fleet (CISD + Liquidity Transfer parallel)
echo   2. Live Market Scanner (live_scanner.py)
echo   3. Dashboard API Server (api_server.py:8000)
echo   4. Frontend Dashboard (Vite: http://localhost:5173)
echo.
echo ==============================================================

cd /d "%~dp0"

:: 1. Launch API Server in a dedicated window
start "API Server [Port 8000]" cmd /k "python api_server.py"

:: 2. Launch Dual Challenge Fleet in a dedicated window
start "Challenge Fleet [CISD + Liquidity Transfer]" cmd /k "python run_challenge_fleet.py"

:: 3. Launch Live Scanner in a dedicated window
start "Live Market Scanner" cmd /k "python live_scanner.py"

:: 4. Launch Frontend Dashboard in a dedicated window
cd frontend
start "Dashboard Frontend [Vite]" cmd /k "npm run dev"

echo.
echo [OK] All 4 trading system instances successfully launched in separate windows!
echo [OK] Dashboard will be accessible at: http://localhost:5173
echo.
pause
