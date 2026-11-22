# Runs the packaged program's AI self-test with the real models (used by CI).
#   pwsh -File packaging/ai_selftest.ps1 -Dist dist -AIDir <folder>
# The program downloads its AI engine and the pinned models exactly as a user
# would, then runs every AI effect on the processor.
param(
    [string]$Dist = "dist",
    [string]$AIDir = (Join-Path $env:TEMP "glitchlab-ai")
)
$ErrorActionPreference = 'Continue'
$env:GLITCHLAB_AI_DIR = $AIDir
$cli = Join-Path (Resolve-Path (Join-Path $Dist "GlitchLab")) "glitchlab-cli.exe"

& $cli --ai-install-engine
if ($LASTEXITCODE -ne 0) { Write-Host "AI engine install failed ($LASTEXITCODE)"; exit 1 }
& $cli --ai-download all
if ($LASTEXITCODE -ne 0) { Write-Host "AI model download failed ($LASTEXITCODE)"; exit 1 }
& $cli --ai-status
& $cli --ai-selftest (Join-Path $Dist "ai-selftest-report.txt") --ai-out (Join-Path $Dist "ai-results")
$code = $LASTEXITCODE
Write-Host "AI self-test exit code: $code"
exit $code
