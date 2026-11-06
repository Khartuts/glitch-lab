#!/usr/bin/env bash
# Build and verify a Linux onedir bundle with the same spec (used to validate
# the packaging configuration on non-Windows build hosts).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
rm -rf build dist
QT_QPA_PLATFORM=offscreen "$PY" -m pytest -q -x || { echo "tests failed"; exit 1; }
"$PY" scripts/make_icon.py
"$PY" -m PyInstaller --noconfirm --clean packaging/glitchlab.spec
APP=dist/GlitchLab
# run the packaged binaries from a different directory and without the dev venv on PATH
cd /
env -i HOME="$HOME" PATH=/usr/bin:/bin QT_QPA_PLATFORM=offscreen GLITCHLAB_CONFIG_DIR="$(mktemp -d)" \
  "$OLDPWD/$APP/glitchlab-cli" --selftest "$OLDPWD/dist/selftest-report.txt"
env -i HOME="$HOME" PATH=/usr/bin:/bin QT_QPA_PLATFORM=offscreen GLITCHLAB_CONFIG_DIR="$(mktemp -d)" \
  "$OLDPWD/$APP/glitchlab-cli" --contact-sheet "$OLDPWD/dist/contact_sheet.png" --overwrite
env -i HOME="$HOME" PATH=/usr/bin:/bin QT_QPA_PLATFORM=offscreen GLITCHLAB_CONFIG_DIR="$(mktemp -d)" \
  "$OLDPWD/$APP/GlitchLab" --smoke-test "$OLDPWD/dist/smoke.png"
echo "Packaged app verified."
