# Launch CTXAgent from this checkout. Refuses to reuse an already-running (possibly stale) backend.
#   .\run.ps1             server on :8765 and the system browser (development)
#   .\run.ps1 --desktop   the native CTXAgent window (see DESKTOP_APP.md)
# If PowerShell refuses this unsigned script (execution policy AllSigned), use run.cmd.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
& $python -m ctxagent.launch @args
exit $LASTEXITCODE
