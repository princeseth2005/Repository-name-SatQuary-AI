@echo off
title SatQuery AI - ISRO Remote Sensing Assistant
echo =====================================================================
echo               SATQUERY AI (ISRO / SIH26167)
echo     Interactive Vision-Language Remote Sensing Assistant
echo =====================================================================
echo.

:: Check python installation
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.10+ and re-run.
    pause
    exit /b 1
)

echo [1/3] Verifying Python environment...
python -c "import fastapi, uvicorn, PIL, cv2, numpy, sqlalchemy, reportlab" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Installing required Python libraries from requirements.txt...
    python -m pip install -r backend\requirements.txt
) else (
    echo [OK] All core dependencies verified.
)

echo [2/3] Verifying sample datasets and database...
python backend\samples\generate_samples.py
python -c "from backend.database import init_db; init_db()"

echo [3/3] Launching SatQuery AI local server...
echo.
echo =====================================================================
echo  SatQuery AI is running at: http://127.0.0.1:8000
echo  Press Ctrl+C in this terminal to stop the server.
echo =====================================================================
echo.

:: Open default browser
start http://127.0.0.1:8000

:: Start Uvicorn
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
pause
