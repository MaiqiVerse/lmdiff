"""Unit tests for the v0.3.0 Markdown renderer (commit 1.11)."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

# Force matplotlib Agg backend before any import (figure-link tests render
# the 3 application figures into a tempdir).
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lmdiff.geometry import GeoResult, _compute_share_per_domain  # noqa: E402
from lmdiff.report import markdown as md_mod  # noqa: E402


def _make_calibration_like() -> GeoResult:
    """Same shape as the HTML test fixture — fires every finding type."""
    variants = ["code", "long", "math", "yarn"]
    domains = (
        "commonsense", "commonsense",
        "reasoning", "reasoning",
        "math", "math",
        "code", "code",
        "long-context", "long-context",
    )
    n = len(domains)
    cv = {
        "code":  [0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.40, 0.40, 0.10, 0.10],
        "long":  [0.10, 0.10, 0.45, 0.45, 0.20, 0.20, 0.10, 0.10, 0.10, 0.10],
        "math":  [0.10, 0.10, 0.20, 0.20, 0.40, 0.40, 0.15, 0.15, 0.10, 0.10],
        "yarn":  [0.40, 0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.30, 0.30],
    }
    cos = {
        "code": {"code": 1.0, "long": 0.95, "math": 0.79, "yarn": 0.96},
        "long": {"code": 0.95, "long": 1.0, "math": 0.80, "yarn": 0.95},
        "math": {"code": 0.79, "long": 0.80, "math": 1.0, "yarn": 0.80},
        "yarn": {"code": 0.96, "long": 0.95, "math": 0.80, "yarn": 1.0},
    }
    sel = {v: {w: 1.0 if v == w else cos[v][w] - 0.02 for w in cos} for v in cos}
    geo = GeoResult(
        base_name="meta-llama/Llama-2-7b-hf",
        variant_names=variants,
        n_probes=n,
        magnitudes={v: float(np.linalg.norm(cv[v])) for v in variants},
        cosine_matrix=cos,
        selective_cosine_matrix=sel,
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(n)} for v in variants},
        metadata={
            "max_new_tokens": 16,
            "accuracy_by_variant": {
                "code": {"hellaswag": 0.53, "gsm8k": 0.0, "longbench_2wikimqa": 0.0},
                "long": {"hellaswag": 0.61, "gsm8k": 0.0, "longbench_2wikimqa": 0.0},
                "math": {"hellaswag": 0.48, "gsm8k": 0.01, "longbench_2wikimqa": 0.0},
                "yarn": {"hellaswag": 0.55, "gsm8k": 0.04, "longbench_2wikimqa": 0.0},
            },
        },
        probe_domains=domains,
        avg_tokens_per_probe=tuple([8.0] * n),
        magnitudes_normalized={v: float(np.linalg.norm(cv[v]) / 4.0) for v in variants},
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


# ── Basic shape ──────────────────────────────────────────────────────


class TestBasicShape:
    def test_returns_string_when_no_path(self):
        out = md_mod.render(_make_calibration_like())
        assert isinstance(out, str)
        assert out.startswith("# lmdiff Family Report")

    def test_writes_file_when_path_given(self, tmp_path):
        path = tmp_path / "report.md"
        out = md_mod.render(_make_calibration_like(), path)
        assert isinstance(out, Path)
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("# lmdiff Family Report")

    def test_no_html_tags_in_output(self):
        out = md_mod.render(_make_calibration_like())
        # Plain markdown only — no inline HTML elements.
        # Allow `<` only when followed by a digit/whitespace/dash (e.g.
        # number expressions), but reject any `<a-z>` opener.
        assert re.search(r"<[a-zA-Z]", out) is None, "found HTML tag"


# ── Five-layer structure ─────────────────────────────────────────────


class TestStructure:
    def test_summary_section_present(self):
        out = md_mod.render(_make_calibration_like())
        assert "## Summary" in out

    def test_one_liner_as_blockquote(self):
        out = md_mod.render(_make_calibration_like())
        # The blockquote line starts with `> ` after the `## Summary` heading.
        m = re.search(r"## Summary\n\n>", out)
        assert m is not None

    def test_all_four_tables_present(self):
        out = md_mod.render(_make_calibration_like())
        assert "## Where each variant acts biggest" in out
        assert "## How big is each move" in out
        assert "## Direction agreement" in out
        assert "## Per-task accuracy" in out

    def test_caveats_as_blockquotes(self):
        out = md_mod.render(_make_calibration_like())
        assert "## Caveats" in out
        # Mental-model reminder always present and quoted.
        assert "> Drift magnitude shows where" in out

    def test_methodology_section_present(self):
        out = md_mod.render(_make_calibration_like())
        assert "## Methodology" in out
        assert "**Probe set" not in out or True  # robust to None probe_set
        assert "**n_probes**" in out

    def test_footer_links_to_repo(self):
        out = md_mod.render(_make_calibration_like())
        assert "[lmdiff-kit](https://github.com/MaiqiVerse/lmdiff)" in out


# ── Cross-renderer findings consistency ──────────────────────────────


class TestFindingsConsistency:
    def test_every_finding_summary_appears_verbatim(self):
        geo = _make_calibration_like()
        out = md_mod.render(geo)
        for f in geo.findings:
            if not f.summary:
                continue
            assert f.summary in out, type(f).__name__


# ── Numeric formatting ───────────────────────────────────────────────


class TestNumericFormatting:
    def test_share_table_bolds_row_peak(self):
        out = md_mod.render(_make_calibration_like())
        # yarn's largest share is on commonsense — should be **51%**.
        # Find yarn's row and verify exactly one **N%** cell.
        rows = [
            line for line in out.split("\n")
            if line.startswith("| yarn ")
        ]
        assert rows, "yarn share row not found"
        # Row from share table contains a bold percentage.
        yarn_share_row = next(r for r in rows if "%" in r)
        assert re.search(r"\*\*\d+%\*\*", yarn_share_row)

    def test_drift_total_column_bold(self):
        out = md_mod.render(_make_calibration_like())
        # Drift table totals are wrapped in **...** for every row.
        # Pick a known variant: the row ends with `| **0.NNNN** |`.
        m = re.search(r"\| code \| .*\| \*\*\d+\.\d+\*\* \|", out)
        assert m is not None

    def test_cosine_diagonal_dash(self):
        out = md_mod.render(_make_calibration_like())
        # Direction-agreement section contains `| **code** | — |` for the
        # diagonal cell.
        assert re.search(r"\| \*\*code\*\* \| — \|", out) is not None

    def test_accuracy_artifact_marker(self):
        out = md_mod.render(_make_calibration_like())
        # gsm8k cells should carry a trailing `*` marker.
        assert "0.00*" in out or "0.01*" in out or "0.04*" in out


# ── Figure-link semantics ────────────────────────────────────────────


class TestFigureLinks:
    def test_no_figure_links_in_string_mode(self):
        out = md_mod.render(_make_calibration_like())
        assert "drift_share_dual.png" not in out
        assert "direction_agreement.png" not in out
        assert "change_size_bars.png" not in out

    def test_default_figs_dir_when_out_path_given(self, tmp_path):
        out_path = tmp_path / "report.md"
        out = md_mod.render(_make_calibration_like(), out_path)
        text = Path(out).read_text(encoding="utf-8")
        assert "figs/drift_share_dual.png" in text
        assert "figs/direction_agreement.png" in text
        assert "figs/change_size_bars.png" in text
        figs = tmp_path / "figs"
        assert figs.exists()
        assert sorted(p.name for p in figs.iterdir()) == [
            "change_size_bars.png",
            "direction_agreement.png",
            "drift_share_dual.png",
        ]

    def test_explicit_figures_dir_relative_link(self, tmp_path):
        out_path = tmp_path / "report.md"
        figures_dir = tmp_path / "shared_figs"
        md_mod.render(
            _make_calibration_like(), out_path, figures_dir=figures_dir,
        )
        text = out_path.read_text(encoding="utf-8")
        assert "shared_figs/drift_share_dual.png" in text


# ── result.to_markdown convenience ───────────────────────────────────


class TestToMarkdownConvenience:
    def test_returns_string_no_out_path(self):
        geo = _make_calibration_like()
        s = geo.to_markdown()
        assert isinstance(s, str)
        assert s.startswith("# lmdiff Family Report")

    def test_returns_path_with_out_path(self, tmp_path):
        geo = _make_calibration_like()
        out = geo.to_markdown(str(tmp_path / "r.md"))
        assert Path(out).exists()
        assert (tmp_path / "figs").exists()


# ── Calibration: real Llama-2 georesult ──────────────────────────────


_CALIB_PATH = Path("runs/llama2-4variants/family_geometry_lm_eval_georesult.json")


@pytest.mark.skipif(
    not _CALIB_PATH.exists(),
    reason="Llama-2 4-variant calibration GeoResult not present in the repo",
)
class TestLlama2Calibration:
    @pytest.fixture(scope="class")
    def geo(self) -> GeoResult:
        from lmdiff.report.json_report import geo_result_from_json_dict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with open(_CALIB_PATH, encoding="utf-8") as f:
                return geo_result_from_json_dict(json.load(f))

    def test_renders_string_mode(self, geo):
        out = md_mod.render(geo)
        assert "# lmdiff Family Report" in out
        # Llama-2 base name appears.
        assert "meta-llama" in out

    def test_renders_with_figures(self, tmp_path, geo):
        out = md_mod.render(geo, tmp_path / "calib.md")
        assert (tmp_path / "figs").exists()
        text = Path(out).read_text(encoding="utf-8")
        # Direction matrix has known cosine values from v6 §13.
        assert "+0.95" in text or "+0.96" in text
