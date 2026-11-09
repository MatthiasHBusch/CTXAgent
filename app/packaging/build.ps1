<#
.SYNOPSIS
  Build the CTXAgent installer: venv -> pip install -> PyInstaller -> Inno Setup.

.DESCRIPTION
  Run from anywhere:  powershell -ExecutionPolicy Bypass -File packaging\build.ps1
  Output:             dist\CTXAgent\ (the app bundle)
                      dist\installer\CTXAgent-Setup-<version>.exe

  The build venv is .venv in the checkout (git-ignored); nothing is installed
  into the Python that creates it. Dependencies are pinned by
  packaging\constraints.txt. Inno Setup 6 (ISCC.exe) is looked up on PATH, in
  %LOCALAPPDATA%\Programs\Inno Setup 6 and in Program Files; install it with
  winget install --id JRSoftware.InnoSetup -e --scope user

.PARAMETER Python
  Interpreter used to create the venv (3.11 recommended). Default: python
.PARAMETER Clean
  Recreate the venv and wipe build\ and dist\ first.
.PARAMETER SkipInstaller
  Stop after the PyInstaller bundle and its self-test.
.PARAMETER Iscc
  Path to ISCC.exe, if it is somewhere unusual.
#>
param(
    [string]$Python = "python",
    [switch]$Clean,
    [switch]$SkipInstaller,
    [string]$Iscc = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Invoke-Step([string]$Title, [scriptblock]$Block) {
    Write-Host "==> $Title" -ForegroundColor Cyan
    # Native tools log to stderr (PyInstaller does all the time); with "Stop"
    # Windows PowerShell 5.1 can turn such a line into a terminating error.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Block } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "$Title failed (exit $LASTEXITCODE)" }
}

$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

if ($Clean) {
    foreach ($dir in @($Venv, (Join-Path $Root "build"), (Join-Path $Root "dist"))) {
        if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force }
    }
}

if (-not (Test-Path -LiteralPath $Py)) {
    Invoke-Step "create venv ($Python)" { & $Python -m venv $Venv }
}
Invoke-Step "upgrade pip" { & $Py -m pip install --disable-pip-version-check --upgrade pip }
Invoke-Step "install dependencies" {
    & $Py -m pip install --disable-pip-version-check -r requirements.txt -c packaging\constraints.txt pywebview pyinstaller
}

$Version = (& $Py -c "from ctxagent.version import __version__; print(__version__)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $Version) { throw "could not read ctxagent/version.py" }
Write-Host "CTXAgent $Version" -ForegroundColor Green

if (-not (Test-Path -LiteralPath "packaging\ctxagent.ico")) {
    Invoke-Step "draw icon" { & $Py packaging\make_icon.py }
}
Invoke-Step "bundle tokenizer data" { & $Py packaging\prepare_tiktoken.py build\tiktoken_cache }
Invoke-Step "PyInstaller" {
    & $Py -m PyInstaller --noconfirm --clean --distpath dist --workpath build\pyinstaller packaging\ctxagent.spec
}

$Cli = Join-Path $Root "dist\CTXAgent\CTXAgentCLI.exe"
Invoke-Step "bundle self-test" { & $Cli --run-module ctxagent.selftest | Out-Host }

$Bundle = Get-ChildItem -LiteralPath (Join-Path $Root "dist\CTXAgent") -Recurse -File | Measure-Object -Sum Length
Write-Host ("bundle: dist\CTXAgent, {0} files, {1:N0} MB" -f $Bundle.Count, ($Bundle.Sum / 1MB))
if ($SkipInstaller) { return }

if (-not $Iscc) {
    $candidates = @(
        (Get-Command iscc -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    $Iscc = $candidates | Select-Object -First 1
}
if (-not $Iscc) {
    throw "Inno Setup 6 not found. Install it (winget install --id JRSoftware.InnoSetup -e --scope user) or pass -Iscc."
}
Invoke-Step "Inno Setup" { & $Iscc /Q "/DMyAppVersion=$Version" packaging\ctxagent.iss }
$Setup = Join-Path $Root "dist\installer\CTXAgent-Setup-$Version.exe"
Write-Host ("installer: {0} ({1:N1} MB)" -f $Setup, ((Get-Item -LiteralPath $Setup).Length / 1MB)) -ForegroundColor Green
