@echo off
rem Launch ctxagent. Works without PowerShell script permissions.
rem Prefers the project venv (.venv); falls back to python on PATH.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m ctxagent.launch
) else (
    python -m ctxagent.launch
)
exit /b %ERRORLEVEL%
