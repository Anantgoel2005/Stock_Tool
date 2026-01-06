@echo off
setlocal

REM Change to this script's directory
cd /d "%~dp0"

REM Activate existing venv (assumes already created)
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please create it first: python -m venv .venv
    exit /b 1
)
call ".venv\Scripts\activate.bat"

echo Starting uvicorn at http://127.0.0.1:8000 ...
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload

endlocal

