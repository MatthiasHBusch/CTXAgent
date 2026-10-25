# Fetch and pin the benchmark's external sources and toolchains (see README.md here).
# Usage: powershell -ExecutionPolicy Bypass -File bench/external/fetch.ps1 [-NoGo] [-NoJs] [-NoOpenCode] [-NoClaudeCode]
param([switch]$NoGo, [switch]$NoJs, [switch]$NoOpenCode, [switch]$NoClaudeCode)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path polyglot-benchmark)) { git clone https://github.com/Aider-AI/polyglot-benchmark.git }
git -C polyglot-benchmark checkout -q 7e0611e

if (-not (Test-Path SciCode)) { git clone https://github.com/scicode-bench/SciCode.git }
git -C SciCode checkout -q e3158ea
if (-not (Test-Path SciCode/eval/data/test_data.h5)) {
  Write-Host "SciCode/eval/data/test_data.h5 is missing: download it as described in SciCode/README.md (section 'Evaluation')."
}

if (-not $NoGo -and -not (Test-Path go/bin/go.exe)) {
  $zip = "go1.27.3.windows-amd64.zip"
  Invoke-WebRequest -Uri "https://go.dev/dl/$zip" -OutFile $zip
  Expand-Archive -Path $zip -DestinationPath . -Force   # creates .\go
  Remove-Item $zip
}

if (-not $NoJs) {
  New-Item -ItemType Directory -Force js_shared | Out-Null
  Copy-Item js_shared.package.json js_shared/package.json -Force
  Push-Location js_shared; npm install --no-audit --no-fund; Pop-Location
}

if (-not $NoOpenCode) {
  New-Item -ItemType Directory -Force opencode | Out-Null
  Copy-Item opencode.package.json opencode/package.json -Force
  Push-Location opencode; npm install --no-audit --no-fund; Pop-Location
}

if (-not $NoClaudeCode) {
  New-Item -ItemType Directory -Force cc | Out-Null
  Push-Location cc
  if (-not (Test-Path package.json)) { npm init -y | Out-Null }
  npm install --no-audit --no-fund "@anthropic-ai/claude-code@2.1.301"
  Pop-Location
}
Write-Host "done. Codex CLI is used from the global install: npm i -g @openai/codex@0.163.0"
