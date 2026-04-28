"""Release-metadata smoke tests for the v0.3.1 release (commit 1.12)."""
from __future__ import annotations

import re
from pathlib import Path

import lmdiff


_ROOT = Path(__file__).resolve().parents[2]


def test_lmdiff_dunder_version_is_0_3_1():
    assert lmdiff.__version__ == "0.3.1"


def test_pyproject_version_is_0_3_1():
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version\s*=\s*"0\.3\.1"\s*$', text, re.MULTILINE), text


def test_changelog_has_v0_3_1_section():
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## \[0\.3\.1\]", text, re.MULTILINE), (
        "CHANGELOG.md missing [0.3.1] heading"
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
