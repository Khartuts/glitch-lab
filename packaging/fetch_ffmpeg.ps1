<#
.SYNOPSIS
  Download a 64-bit static LGPL FFmpeg build for bundling with Glitch Lab.

.DESCRIPTION
  Downloads the archive and the release's checksums.sha256 from BtbN/FFmpeg-Builds,
  verifies the SHA-256, and copies bin\ffmpeg.exe and LICENSE.txt into packaging\ffmpeg\.
  "latest" is a moving release: for reproducible builds pass -Release with a dated
  tag (e.g. autobuild-2026-10-31-12-55) and record the hash printed below.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\fetch_ffmpeg.ps1
  powershell -ExecutionPolicy Bypass -File packaging\fetch_ffmpeg.ps1 -Asset ffmpeg-n9.0-latest-win64-lgpl-9.0.zip
#>
param(
    [string]$Release = "latest",
    [string]$Asset = "ffmpeg-n8.1-latest-win64-lgpl-8.1.zip"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $PSScriptRoot "ffmpeg"
$tmp = Join-Path $env:TEMP ("glitchlab-ffmpeg-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $tmp, $dest | Out-Null
$base = "https://github.com/BtbN/FFmpeg-Builds/releases/download/$Release"
try {
    Write-Host "Downloading $base/$Asset"
    Invoke-WebRequest "$base/$Asset" -OutFile (Join-Path $tmp $Asset) -UseBasicParsing
    Invoke-WebRequest "$base/checksums.sha256" -OutFile (Join-Path $tmp "checksums.sha256") -UseBasicParsing
    $expected = (Get-Content (Join-Path $tmp "checksums.sha256") | Where-Object { $_ -match [regex]::Escape($Asset) } |
                 ForEach-Object { ($_ -split "\s+")[0] }) | Select-Object -First 1
    $actual = (Get-FileHash (Join-Path $tmp $Asset) -Algorithm SHA256).Hash.ToLower()
    if (-not $expected) { throw "No checksum for $Asset in checksums.sha256" }
    if ($actual -ne $expected.ToLower()) { throw "SHA-256 mismatch: expected $expected got $actual" }
    Write-Host "SHA-256 OK: $actual"
    Expand-Archive (Join-Path $tmp $Asset) -DestinationPath $tmp -Force
    $exe = Get-ChildItem $tmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    $lic = Get-ChildItem $tmp -Recurse -Filter LICENSE.txt | Select-Object -First 1
    Copy-Item $exe.FullName (Join-Path $dest "ffmpeg.exe") -Force
    if ($lic) { Copy-Item $lic.FullName (Join-Path $dest "FFMPEG_LICENSE.txt") -Force }
    "Source: $base/$Asset`r`nSHA-256: $actual`r`nFFmpeg source code: https://ffmpeg.org/download.html and https://github.com/BtbN/FFmpeg-Builds" |
        Set-Content (Join-Path $dest "FFMPEG_SOURCE.txt")
    & (Join-Path $dest "ffmpeg.exe") -hide_banner -version | Select-Object -First 1
    Write-Host "ffmpeg.exe placed in $dest"
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
