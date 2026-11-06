@echo off
rem Convenience wrapper: build Glitch Lab with FFmpeg and a zip package.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" -FetchFFmpeg %*
