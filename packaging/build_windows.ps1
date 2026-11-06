<#
.SYNOPSIS
  Build, test and package Glitch Lab for Windows 10/11 x64.

.DESCRIPTION
  1. Creates .venv-win with Python 3.13 (64-bit) and installs pinned dependencies.
  2. Optionally downloads FFmpeg (-FetchFFmpeg) into packaging\ffmpeg\ (SHA-256 verified).
  3. Runs the automated test suite (engine + offscreen UI) - Sonify tests use the
     same ffmpeg.exe that will be shipped.
  4. Builds dist\GlitchLab\ with PyInstaller (no Python needed on target machines).
  5. Verifies the PACKAGED app separately from the dev environment:
       glitchlab-cli.exe --selftest, --contact-sheet, and GlitchLab.exe --smoke-test.
  6. Zips dist\GlitchLab and, with -Installer, builds an Inno Setup installer.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -FetchFFmpeg -Installer
#>
param(
    [string]$Python = "",            # explicit python.exe (e.g. from actions/setup-python)
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.13",
    [switch]$FetchFFmpeg,
    [switch]$SkipTests,
    [switch]$Installer,
    [switch]$Headless                # run the GUI smoke test with the offscreen Qt platform (CI)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$venv = Join-Path $root ".venv-win"
$py = Join-Path $venv "Scripts\python.exe"
$version = "1.0.0"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Magenta }
function Check([string]$what) {
    # Call directly after a native command: fail the build on a non-zero exit code.
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

Step "Python environment"
if (-not (Test-Path $py)) {
    if ($Python) { & $Python -m venv $venv } else { & $PythonLauncher "-$PythonVersion" -m venv $venv }
    Check "Creating the virtual environment"
}
& $py -c "import struct,sys; assert struct.calcsize('P')==8, '64-bit Python required'; print(sys.version)"; Check "64-bit Python check"
& $py -m pip install --upgrade pip; Check "pip upgrade"
& $py -m pip install -r requirements-dev.txt; Check "Installing pinned dependencies"
& $py -m pip install --no-deps -e .; Check "Installing Glitch Lab (editable)"

$ffexe = Join-Path $PSScriptRoot "ffmpeg\ffmpeg.exe"
if ($FetchFFmpeg) { Step "FFmpeg"; & (Join-Path $PSScriptRoot "fetch_ffmpeg.ps1") }
if (Test-Path $ffexe) {
    & $ffexe -hide_banner -version; Check "Running the bundled ffmpeg.exe"
} else {
    Write-Warning "packaging\ffmpeg\ffmpeg.exe not found: the package will work, but Sonify needs FFmpeg on PATH."
}

if (-not $SkipTests) {
    Step "Automated tests"
    $env:QT_QPA_PLATFORM = "offscreen"
    if (Test-Path $ffexe) { $env:GLITCHLAB_FFMPEG = $ffexe; $env:GLITCHLAB_REQUIRE_FFMPEG = "1" }
    & $py -m pytest -q -rs; Check "Automated tests"
    Remove-Item Env:QT_QPA_PLATFORM
    Remove-Item Env:GLITCHLAB_FFMPEG -ErrorAction SilentlyContinue
    Remove-Item Env:GLITCHLAB_REQUIRE_FFMPEG -ErrorAction SilentlyContinue
}

Step "Icon"
& $py scripts\make_icon.py; Check "Icon generation"

Step "PyInstaller"
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
& $py -m PyInstaller --noconfirm --clean packaging\glitchlab.spec; Check "PyInstaller"

Step "Verify packaged application (independent of .venv-win)"
$app = Join-Path $root "dist\GlitchLab"
$cli = Join-Path $app "glitchlab-cli.exe"
$env:PYTHONPATH = ""; $env:PYTHONHOME = ""
$env:GLITCHLAB_CONFIG_DIR = Join-Path $env:TEMP ("glitchlab-verify-" + [guid]::NewGuid())
& $cli --version; Check "Packaged glitchlab-cli --version"
& $cli --info; Check "Packaged glitchlab-cli --info"
& $cli --selftest (Join-Path $root "dist\selftest-report.txt")
$selftestExit = $LASTEXITCODE
Get-Content (Join-Path $root "dist\selftest-report.txt")
if ($selftestExit -ne 0) { throw "Packaged self-test failed - see dist\selftest-report.txt" }
if ((Test-Path $ffexe) -and -not (Select-String -Path (Join-Path $root "dist\selftest-report.txt") -Pattern "bundled with Glitch Lab" -Quiet)) {
    throw "The packaged app did not pick up its bundled ffmpeg.exe"
}
if ((Test-Path $ffexe) -and (Select-String -Path (Join-Path $root "dist\selftest-report.txt") -Pattern "^SKIP" -Quiet)) {
    throw "The packaged self-test skipped checks although FFmpeg is bundled - see dist\selftest-report.txt"
}
& $cli --contact-sheet (Join-Path $root "dist\contact_sheet.png") --overwrite; Check "Packaged contact sheet"
if ($Headless) { $env:QT_QPA_PLATFORM = "offscreen" }
$p = Start-Process -FilePath (Join-Path $app "GlitchLab.exe") -ArgumentList "--smoke-test", ('"' + (Join-Path $root "dist\smoke.png") + '"') -PassThru
if (-not $p.WaitForExit(300000)) { $p.Kill(); throw "GUI smoke test timed out" }
if ($p.ExitCode -ne 0) { throw "GUI smoke test failed (exit $($p.ExitCode))" }
if ($Headless) { Remove-Item Env:QT_QPA_PLATFORM }
Remove-Item Env:GLITCHLAB_CONFIG_DIR
Write-Host "Packaged app verified: dist\selftest-report.txt, dist\contact_sheet.png, dist\smoke.png"

Step "Archive"
$zip = Join-Path $root "dist\GlitchLab-$version-win64.zip"
Compress-Archive -Path $app -DestinationPath $zip -Force
Write-Host "Portable package: $zip"

if ($Installer) {
    Step "Installer (Inno Setup 6)"
    $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $iscc)) {
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            & choco install innosetup -y --no-progress; Check "Installing Inno Setup"
        } else {
            throw "Inno Setup 6 not found at $iscc (install it with: winget install JRSoftware.InnoSetup)"
        }
    }
    & $iscc (Join-Path $PSScriptRoot "installer.iss"); Check "Inno Setup compiler"
    Write-Host "Installer: dist\GlitchLab-$version-setup-win64.exe"
}
