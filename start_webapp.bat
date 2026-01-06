@echo off
setlocal

REM Change to the directory where this script lives
cd /d "%~dp0"

REM Create venv if missing
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call ".venv\Scripts\activate.bat"

REM Install/refresh dependencies quietly
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt

REM Start the FastAPI app
echo Starting server at http://127.0.0.1:8000 ...
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload

endlocal