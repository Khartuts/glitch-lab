"""All version strings used for packaging must agree."""

import re
from pathlib import Path

from glitchlab import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_versions_consistent():
    pyproject = re.search(r'^version = "(.+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    iss = re.search(r'#define AppVersion "(.+)"', (ROOT / "packaging" / "installer.iss").read_text()).group(1)
    info = (ROOT / "packaging" / "version_info.txt").read_text()
    file_version = re.search(r"StringStruct\('FileVersion', '(.+?)'\)", info).group(1)
    tup = re.search(r"filevers=\((\d+), (\d+), (\d+), \d+\)", info).groups()
    assert __version__ == pyproject == iss == file_version == ".".join(tup)
