"""Unit tests for the v0.3.0 HTML self-contained renderer (commit 1.10)."""
from __future__ import annotations

import base64
import json
import warnings
from html.parser import HTMLParser
from pathlib import Path

# Force a non-interactive matplotlib backend before any import.
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lmdiff.geometry import GeoResult, _compute_share_per_domain  # noqa: E402
from lmdiff.report import html as html_mod  # noqa: E402


# ── Fixture ──────────────────────────────────────────────────────────


def _make_calibration_like() -> GeoResult:
    """4-variant × 5-domain GeoResult mirroring the Llama-2 calibration shape.

    Triggers cluster + outlier + specialization peaks + accuracy artifact +
    base-accuracy-missing findings.
    """
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
    sel = {
        v: {w: 1.0 if v == w else cos[v][w] - 0.02 for w in cos} for v in cos
    }
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


# ── Minimal HTML well-formedness checker ─────────────────────────────


class _MinimalHTMLValidator(HTMLParser):
    """Parse-only check using stdlib html.parser. Asserts DOCTYPE +
    balanced non-void tags."""

    _VOID = {"br", "hr", "img", "meta", "input", "link", "source", "track"}

    def __init__(self):
        super().__init__()
        self.errors: list[str] = []
        self.stack: list[str] = []
        self.has_doctype = False

    def handle_decl(self, decl: str):
        if decl.lower().startswith("doctype html"):
            self.has_doctype = True

    def handle_starttag(self, tag, attrs):
        if tag in self._VOID:
            return
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack:
            self.errors.append(f"closing {tag} with empty stack")
            return
        if self.stack[-1] != tag:
            self.errors.append(
                f"mismatched closing {tag} (expected {self.stack[-1]})"
            )
            return
        self.stack.pop()


# ── Basic shape ──────────────────────────────────────────────────────


class TestBasicShape:
    def test_returns_string_when_no_path(self):
        out = html_mod.render(_make_calibration_like())
        assert isinstance(out, str)
        assert out.startswith("<!DOCTYPE html>")
        assert out.rstrip().endswith("</html>")

    def test_writes_file_when_path_given(self, tmp_path):
        out = html_mod.render(_make_calibration_like(), tmp_path / "r.html")
        assert isinstance(out, Path)
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_html_well_formed(self):
        out = html_mod.render(_make_calibration_like())
        v = _MinimalHTMLValidator()
        v.feed(out)
        assert v.has_doctype
        assert v.errors == [], v.errors
        assert v.stack == [], v.stack


# ── Embed-images modes ───────────────────────────────────────────────


class TestEmbedImagesModes:
    def test_default_embeds_base64(self, tmp_path):
        out = html_mod.render(
            _make_calibration_like(), tmp_path / "embed.html",
        )
        text = out.read_text(encoding="utf-8")
        # All <img> tags use data:image/png;base64 — no relative paths.
        assert "data:image/png;base64," in text
        assert 'src="figs/' not in text
        # No figs/ subdir created.
        assert not (tmp_path / "figs").exists()
        # Reasonable size for an embedded report.
        size = out.stat().st_size
        assert 200_000 < size < 3_000_000, size

    def test_external_creates_figs_dir(self, tmp_path):
        out = html_mod.render(
            _make_calibration_like(), tmp_path / "ext.html",
            embed_images=False,
        )
        text = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," not in text
        assert 'src="figs/drift_share_dual.png"' in text
        assert 'src="figs/direction_agreement.png"' in text
        assert 'src="figs/change_size_bars.png"' in text
        figs = tmp_path / "figs"
        assert figs.exists()
        children = sorted(p.name for p in figs.iterdir())
        assert children == [
            "change_size_bars.png",
            "direction_agreement.png",
            "drift_share_dual.png",
        ]
        # External-mode HTML stays small.
        assert out.stat().st_size < 100_000

    def test_external_requires_out_path(self):
        with pytest.raises(ValueError, match="embed_images=False requires out_path"):
            html_mod.render(_make_calibration_like(), embed_images=False)

    def test_external_includes_co_location_comment(self, tmp_path):
        out = html_mod.render(
            _make_calibration_like(), tmp_path / "ext.html",
            embed_images=False,
        )
        text = out.read_text(encoding="utf-8")
        assert "<!-- This HTML references PNG files" in text


# ── Findings consistency (cross-renderer) ────────────────────────────


class TestFindingsConsistency:
    def test_every_finding_summary_appears_verbatim(self):
        geo = _make_calibration_like()
        out = html_mod.render(geo)
        for f in geo.findings:
            if not f.summary:
                continue
            from html import escape as _e
            # HTML-escape the summary for comparison since the renderer
            # does the same when emitting it.
            assert _e(f.summary) in out, type(f).__name__


# ── Theme + script ──────────────────────────────────────────────────


class TestThemeAndScript:
    def test_theme_toggle_button_present(self):
        out = html_mod.render(_make_calibration_like())
        assert 'id="theme-toggle"' in out
        assert "lmdiff-theme" in out  # localStorage key

    def test_default_theme_attribute(self):
        out = html_mod.render(_make_calibration_like())
        assert 'data-theme="auto"' in out

    def test_theme_override_dark(self):
        out = html_mod.render(_make_calibration_like(), theme="dark")
        assert 'data-theme="dark"' in out

    def test_invalid_theme_rejected(self):
        with pytest.raises(ValueError, match="theme must be"):
            html_mod.render(_make_calibration_like(), theme="rainbow")

    def test_light_and_dark_css_rules_present(self):
        out = html_mod.render(_make_calibration_like())
        assert ":root {" in out
        assert '[data-theme="dark"]' in out
        assert "prefers-color-scheme: dark" in out

    def test_print_stylesheet_present(self):
        out = html_mod.render(_make_calibration_like())
        assert "@media print" in out


# ── Section presence ────────────────────────────────────────────────


class TestSections:
    def test_summary_one_liner_present(self):
        out = html_mod.render(_make_calibration_like())
        assert 'class="one-liner"' in out
        assert "<h2>Summary</h2>" in out

    def test_three_figure_sections_with_headings(self):
        out = html_mod.render(_make_calibration_like())
        assert "Where each variant acts biggest</h2>" in out
        assert "Direction agreement</h2>" in out
        assert "How big is each move</h2>" in out

    def test_numeric_tables_present(self):
        out = html_mod.render(_make_calibration_like())
        assert "<h3>Where each variant acts biggest</h3>" in out
        assert "<h3>How big is each move</h3>" in out
        assert "<h3>Direction agreement</h3>" in out
        assert "<h3>Per-task accuracy</h3>" in out

    def test_caveats_section_with_mental_model_reminder(self):
        out = html_mod.render(_make_calibration_like())
        assert '<h2>Caveats</h2>' in out
        assert "Drift magnitude shows where" in out

    def test_methodology_collapsed_by_default(self):
        out = html_mod.render(_make_calibration_like())
        # <details> without `open` attribute → collapsed.
        assert '<details class="methodology">' in out
        assert '<details class="methodology" open' not in out

    def test_footer_links_to_repo(self):
        out = html_mod.render(_make_calibration_like())
        assert 'href="https://github.com/MaiqiVerse/lmdiff"' in out


# ── Accuracy artifact marker ────────────────────────────────────────


class TestAccuracyArtifactMarker:
    def test_marker_appears_on_artifact_cells(self):
        geo = _make_calibration_like()
        out = html_mod.render(geo)
        assert 'class="cell-artifact-marker"' in out


# ── result.to_html convenience method ────────────────────────────────


class TestGeoResultToHtmlConvenience:
    def test_to_html_returns_string_by_default(self):
        geo = _make_calibration_like()
        s = geo.to_html()
        assert isinstance(s, str)
        assert s.startswith("<!DOCTYPE html>")

    def test_to_html_writes_file(self, tmp_path):
        geo = _make_calibration_like()
        out = geo.to_html(str(tmp_path / "r.html"))
        assert Path(out).exists()

    def test_to_html_external_mode(self, tmp_path):
        geo = _make_calibration_like()
        out = geo.to_html(str(tmp_path / "r.html"), embed_images=False)
        assert (tmp_path / "figs").exists()
        text = Path(out).read_text(encoding="utf-8")
        assert "data:image/png;base64," not in text


# ── Calibration: render the real Llama-2 georesult ──────────────────


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

    def test_renders_embed_mode(self, tmp_path, geo):
        out = html_mod.render(geo, tmp_path / "calib.html")
        assert out.exists()
        v = _MinimalHTMLValidator()
        v.feed(out.read_text(encoding="utf-8"))
        assert v.has_doctype
        assert v.errors == [], v.errors

    def test_renders_external_mode(self, tmp_path, geo):
        out = html_mod.render(geo, tmp_path / "calib.html", embed_images=False)
        assert out.exists()
        figs = tmp_path / "figs"
        assert sorted(p.name for p in figs.iterdir()) == [
            "change_size_bars.png",
            "direction_agreement.png",
            "drift_share_dual.png",
        ]
