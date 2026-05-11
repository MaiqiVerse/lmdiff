"""Release-metadata smoke tests — pinned to the current release.

Updated for v0.4.0 (PR #15 backend cutover). Bumping the version
in ``pyproject.toml`` / ``lmdiff/__init__.py`` requires updating
the pinned strings here in the same commit; the previous v0.3.2
version is still checked as a CHANGELOG history entry.
"""
from __future__ import annotations

import re
from pathlib import Path

import lmdiff


_ROOT = Path(__file__).resolve().parents[2]
_CURRENT_VERSION = "0.4.0"


def test_lmdiff_dunder_version_is_current():
    assert lmdiff.__version__ == _CURRENT_VERSION


def test_pyproject_version_is_current():
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pattern = rf'^version\s*=\s*"{re.escape(_CURRENT_VERSION)}"\s*$'
    assert re.search(pattern, text, re.MULTILINE), text


def test_changelog_has_current_section():
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = rf"^## \[{re.escape(_CURRENT_VERSION)}\]"
    assert re.search(pattern, text, re.MULTILINE), (
        f"CHANGELOG.md missing [{_CURRENT_VERSION}] heading"
    )


def test_changelog_retains_v0_3_2_history():
    """Past release headings must remain in the CHANGELOG so the
    history isn't lost on each bump."""
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## \[0\.3\.2\]", text, re.MULTILINE), (
        "CHANGELOG.md missing historical [0.3.2] heading"
    )


def test_migration_guide_exists_and_covers_required_sections():
    path = _ROOT / "docs" / "migration" / "v02-to-v03.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    # Required sections per the release spec.
    for required in (
        "What changed",
        "Quick mapping table",
        "Deprecation timeline",
        "Configuration class deep-dive",
        "Custom Engine integration",
        "Reporting your migration experience",
    ):
        assert required in text, f"migration guide missing section: {required}"
