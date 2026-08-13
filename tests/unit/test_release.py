"""Release-metadata smoke tests — pinned to the current release.

Updated for v0.4.1 (PR #19 measurement validity + pdn correction).
Bumping the version in ``pyproject.toml`` / ``lmdiff/__init__.py``
requires updating the pinned strings here in the same commit; older
versions are still checked as CHANGELOG history entries.

``test_changelog_current_section_is_dated`` exists because the v0.4.1
section carried its drafting date (2026-05-12) for three months while
the work finished — the heading is written early and nothing re-reads
it at tag time.
"""
from __future__ import annotations

import re
from pathlib import Path

import lmdiff


_ROOT = Path(__file__).resolve().parents[2]
_CURRENT_VERSION = "0.4.1"


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


def test_changelog_current_section_is_dated():
    """The current section must carry a real ISO date, not the
    placeholder-ish drafting date it was written with.

    Guards the failure this test was added for: the v0.4.1 heading was
    authored as ``2026-05-12`` and stayed that way for three months
    while the release finished. Nothing else re-reads the date at tag
    time, so it silently ships wrong.
    """
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = rf"^## \[{re.escape(_CURRENT_VERSION)}\] - (\d{{4}}-\d{{2}}-\d{{2}})\s*$"
    m = re.search(pattern, text, re.MULTILINE)
    assert m, (
        f"CHANGELOG.md [{_CURRENT_VERSION}] heading must end with an ISO date"
    )


def test_changelog_retains_release_history():
    """Past release headings must remain in the CHANGELOG so the
    history isn't lost on each bump."""
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for past in ("0.4.0", "0.3.2"):
        assert re.search(rf"^## \[{re.escape(past)}\]", text, re.MULTILINE), (
            f"CHANGELOG.md missing historical [{past}] heading"
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
