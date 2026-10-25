# Launch ctxagent. Refuse to reuse an already-running (possibly stale) backend.
# If PowerShell refuses this unsigned script (execution policy AllSigned), use run.cmd.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
& $python -m ctxagent.launch
exit $LASTEXITCODE
