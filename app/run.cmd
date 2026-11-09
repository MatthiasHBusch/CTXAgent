@echo off
rem Launch CTXAgent from this checkout. Works without PowerShell script permissions.
rem   run.cmd             server on :8765 and the system browser (development)
rem   run.cmd --desktop   the native CTXAgent window (see DESKTOP_APP.md)
rem Prefers the project venv (.venv); falls back to python on PATH.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m ctxagent.launch %*
) else (
    python -m ctxagent.launch %*
)
exit /b %ERRORLEVEL%
