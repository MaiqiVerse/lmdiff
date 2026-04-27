"""Unit tests for the v0.3.0 ``lmdiff.report`` skeleton (commit 1.5).

Covers:

* The ``Renderer`` Protocol — each of the 5 renderer modules conforms.
* ``lmdiff.report`` import does NOT load matplotlib / heavy deps.
* ``lmdiff.report.render(result, channel=...)`` dispatches correctly.
* Convenience methods on ``GeoResult`` — ``print``, ``to_html``,
  ``to_markdown``, ``save``, ``figures`` — delegate to the right renderer.
* ``lmdiff.load_result`` round-trips with ``result.save`` and accepts both
  v4 (with DeprecationWarning) and v5 JSON.
* HTML output is valid HTML5 (parseable by ``html.parser``).
* Unknown channel raises ``ValueError`` listing valid channels.
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from html.parser import HTMLParser
from pathlib import Path

import pytest

from lmdiff.geometry import GeoResult, _compute_share_per_domain
from lmdiff.report import _protocols, _pipeline


def _make_geo() -> GeoResult:
    n = 4
    cv = {
        "yarn": [3.0, 4.0, 0.0, 0.0],
        "long": [0.0, 0.0, 6.0, 8.0],
    }
    geo = GeoResult(
        base_name="base",
        variant_names=list(cv.keys()),
        n_probes=n,
        magnitudes={
            "yarn": float((9.0 + 16.0) ** 0.5),
            "long": float((36.0 + 64.0) ** 0.5),
        },
        cosine_matrix={
            v: {w: 1.0 if v == w else 0.0 for w in cv} for v in cv
        },
        change_vectors=cv,
        per_probe={v: {f"p{i}": cv[v][i] for i in range(n)} for v in cv},
        metadata={},
        probe_domains=("a", "a", "b", "b"),
        avg_tokens_per_probe=tuple([8.0] * n),
    )
    geo.share_per_domain = _compute_share_per_domain(geo)
    return geo


# ── Protocol conformance ──────────────────────────────────────────────


class TestRendererProtocolConformance:
    def test_terminal_module_conforms(self):
        from lmdiff.report import terminal
        assert isinstance(terminal, _protocols.Renderer)

    def test_markdown_module_conforms(self):
        from lmdiff.report import markdown
        assert isinstance(markdown, _protocols.Renderer)

    def test_html_module_conforms(self):
        from lmdiff.report import html
        assert isinstance(html, _protocols.Renderer)

    def test_json_module_conforms(self):
        from lmdiff.report import json_report
        assert isinstance(json_report, _protocols.Renderer)

    def test_figures_module_conforms(self):
        from lmdiff.report import figures
        assert isinstance(figures, _protocols.Renderer)


# ── Lazy import contract ──────────────────────────────────────────────


class TestLazyImport:
    def test_report_import_no_matplotlib(self):
        if "matplotlib" in sys.modules:
            pytest.skip("matplotlib already loaded by previous test")
        import lmdiff.report  # noqa: F401
        assert "matplotlib" not in sys.modules

    def test_report_import_no_torch(self):
        if "torch" in sys.modules:
            pytest.skip("torch already loaded")
        import lmdiff.report  # noqa: F401
        assert "torch" not in sys.modules


# ── Pipeline dispatch ─────────────────────────────────────────────────


class TestPipelineDispatch:
    def test_unknown_channel_raises_with_valid_list(self):
        geo = _make_geo()
        with pytest.raises(ValueError, match="unknown render channel"):
            _pipeline.render(geo, channel="bogus")

    def test_terminal_channel_returns_str(self):
        geo = _make_geo()
        out = _pipeline.render(geo, channel="terminal", file=io.StringIO())
        assert isinstance(out, str)
        # The v0.3.0 5-layer renderer (commit 1.7) opens with the
        # banner "Family experiment:" rather than the v0.3.0-rc stub's
        # "GeoResult(...)" marker.
        assert "Family experiment" in out

    def test_markdown_channel_returns_md_string(self):
        geo = _make_geo()
        out = _pipeline.render(geo, channel="markdown")
        assert isinstance(out, str)
        assert out.startswith("# lmdiff:")
        assert "| variant |" in out

    def test_html_channel_returns_html_string(self):
        geo = _make_geo()
        out = _pipeline.render(geo, channel="html")
        assert out.startswith("<!DOCTYPE html>")
        assert "</html>" in out

    def test_json_channel_returns_dict(self):
        geo = _make_geo()
        out = _pipeline.render(geo, channel="json")
        assert isinstance(out, dict)
        assert out["schema_version"] == "5"

    def test_figures_channel_applied_tier_returns_list(self, tmp_path):
        # Commit 1.9 default: tier='applied' returns list[Path] of 3 PNGs.
        geo = _make_geo()
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg")
        except ImportError:
            pytest.skip("matplotlib not available; this test requires it")
        rendered = _pipeline.render(
            geo,
            channel="figures",
            out_dir=tmp_path,
            variant_order=["yarn", "long"],
        )
        assert isinstance(rendered, list)
        assert len(rendered) == 3
        for p in rendered:
            assert Path(p).exists()

    def test_figures_channel_paper_tier_still_returns_dict(self, tmp_path):
        # tier='paper' delegates to v0.2.x plot_family_figures.
        geo = _make_geo()
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg")
        except ImportError:
            pytest.skip("matplotlib not available; this test requires it")
        rendered = _pipeline.render(
            geo,
            channel="figures",
            out_dir=tmp_path,
            tier="paper",
            variant_order=["yarn", "long"],
        )
        assert isinstance(rendered, dict)
        assert "cosine_raw" in rendered


# ── build_tables ─────────────────────────────────────────────────────


class TestBuildTables:
    def test_keys_present_even_when_empty(self):
        geo = _make_geo()
        tables = _pipeline.build_tables(geo)
        assert {
            "magnitudes", "magnitudes_norm", "cosine", "selective_cosine",
            "share", "zscore", "accuracy",
        } <= tables.keys()

    def test_share_populated_from_v5_field(self):
        geo = _make_geo()
        tables = _pipeline.build_tables(geo)
        assert tables["share"] == geo.share_per_domain


# ── HTML output well-formedness ───────────────────────────────────────


class _MinimalHTMLValidator(HTMLParser):
    def __init__(self):
        super().__init__()
        self.errors: list[str] = []
        self.stack: list[str] = []
        self.has_doctype = False

    def handle_decl(self, decl):
        if decl.lower().startswith("doctype html"):
            self.has_doctype = True

    def handle_starttag(self, tag, attrs):
        # Treat void elements as self-closing
        if tag in ("br", "hr", "img", "meta", "input", "link"):
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


class TestHtmlValidity:
    def test_html_renders_valid_doctype_and_structure(self):
        geo = _make_geo()
        out = _pipeline.render(geo, channel="html")
        v = _MinimalHTMLValidator()
        v.feed(out)
        assert v.has_doctype
        assert v.errors == [], v.errors
        assert v.stack == [], v.stack


# ── Convenience methods on GeoResult ──────────────────────────────────


class TestGeoResultConvenience:
    def test_print_writes_to_stdout(self, capsys):
        geo = _make_geo()
        geo.print()
        captured = capsys.readouterr()
        # The v0.3.0 5-layer renderer (commit 1.7) opens with the banner
        # "Family experiment:" and ends with the closing rule.
        assert "Family experiment" in captured.out
        assert "Headlines" in captured.out
        assert "See also" in captured.out

    def test_to_html_returns_path_when_out_path_given(self, tmp_path):
        # Commit 1.10 contract: with out_path, returns the Path written to;
        # without, returns the HTML string.
        geo = _make_geo()
        path = tmp_path / "out.html"
        out = geo.to_html(str(path))
        assert isinstance(out, Path)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<!DOCTYPE html>")

    def test_to_html_returns_string_when_no_out_path(self, tmp_path):
        geo = _make_geo()
        out = geo.to_html()
        assert isinstance(out, str)
        assert out.startswith("<!DOCTYPE html>")

    def test_to_markdown_returns_md_and_writes_file(self, tmp_path):
        geo = _make_geo()
        path = tmp_path / "out.md"
        out = geo.to_markdown(str(path))
        assert out.startswith("# lmdiff:")
        assert path.exists()

    def test_save_writes_v5_json(self, tmp_path):
        geo = _make_geo()
        path = tmp_path / "out.json"
        geo.save(str(path))
        assert path.exists()
        d = json.loads(path.read_text(encoding="utf-8"))
        assert d["schema_version"] == "5"
        assert "share_per_domain" in d


# ── load_result ───────────────────────────────────────────────────────


class TestLoadResultRoundTrip:
    def test_save_then_load_equals(self, tmp_path):
        from lmdiff import load_result
        geo = _make_geo()
        path = tmp_path / "result.json"
        geo.save(str(path))
        restored = load_result(str(path))
        assert restored.variant_names == geo.variant_names
        assert restored.share_per_domain == geo.share_per_domain
        assert restored.magnitudes == geo.magnitudes

    def test_load_v4_emits_warning(self, tmp_path):
        from lmdiff import load_result
        geo = _make_geo()
        # Hand-write a v4 payload.
        from lmdiff.report.json_report import to_json_dict
        v4 = to_json_dict(geo)
        v4.pop("share_per_domain")
        v4["schema_version"] = "4"
        path = tmp_path / "v4.json"
        path.write_text(json.dumps(v4), encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            restored = load_result(str(path))
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        # In-memory result is v5-shaped after load.
        assert restored.share_per_domain
