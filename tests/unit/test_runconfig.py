"""Run-configuration schema and emitter — v0.4.3.

Specification: ``docs/internal/v043_runconfig_design.md``, and the
settled decisions it builds on in PHASE_PLAN Update 7 AA.
"""
from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from lmdiff import Config, DecodeSpec  # noqa: E402
from lmdiff._runconfig import (  # noqa: E402
    PROVENANCE_KEY,
    RUNCONFIG_SCHEMA_VERSION,
    RunConfigEmitError,
    build_run_config,
    emit_run_config_yaml,
)


def _doc(**over):
    kwargs = dict(
        base=Config(model="base-model"),
        variants={
            "plain": Config(model="v-plain"),
            "sampled": Config(
                model="v-sampled",
                decode=DecodeSpec(strategy="sample", temperature=1.5),
            ),
        },
        probes="v01",
        n_probes=100,
        metrics=["bd", "drift"],
        max_new_tokens=16,
        task_overrides={"gsm8k": {"max_new_tokens": 256}},
        seed=42,
        min_valid_fraction=0.5,
        run_id="2026-08-20T00:00:00Z",
    )
    kwargs.update(over)
    return build_run_config(**kwargs)


# ── schema shape (§3.1) ──────────────────────────────────────────────


class TestSchemaShape:
    def test_required_keys_present(self):
        d = _doc()
        for key in (
            "lmdiff_schema", "base", "variants", "probes", "n_probes",
            "metrics", "max_new_tokens", "task_overrides", "seed",
            "min_valid_fraction", "reproducible", PROVENANCE_KEY,
        ):
            assert key in d, key

    def test_schema_version_is_independent_of_geo_schema(self):
        """Three version numbers, deliberately (§10.5). This one versions
        an *input* grammar; geo_schema versions an *output* shape."""
        from lmdiff.report.json_report import SCHEMA_VERSION
        d = _doc()
        assert d["lmdiff_schema"] == RUNCONFIG_SCHEMA_VERSION
        assert str(d["lmdiff_schema"]) != SCHEMA_VERSION

    def test_reproducible_ships_true_with_no_occupant(self):
        """AA.5. §1.1 confirmed nothing on Config is unserializable
        today, which makes this exactly the field an implementer drops
        as 'not needed yet'."""
        assert _doc()["reproducible"] is True


# ── variant shorthand (§3.1, §3.2) ───────────────────────────────────


class TestVariantShorthand:
    """A bare model-id when the Config is all-default apart from
    ``model``; an expanded mapping otherwise.

    Not cosmetic. §3.2 shows the seven-variant document is unreadable
    without it — five near-identical seven-line ``decode`` blocks bury
    the two variants that actually differ.
    """

    def test_plain_config_renders_as_a_bare_string(self):
        assert _doc()["variants"]["plain"] == "v-plain"

    def test_non_default_config_renders_as_a_mapping(self):
        block = _doc()["variants"]["sampled"]
        assert isinstance(block, dict)
        assert block["model"] == "v-sampled"
        assert block["decode"]["strategy"] == "sample"

    def test_system_prompt_forces_expansion(self):
        d = _doc(variants={"sp": Config(model="m", system_prompt="be terse")})
        assert d["variants"]["sp"]["system_prompt"] == "be terse"

    def test_name_is_dropped_from_the_block(self):
        """The mapping key already carries the name; repeating it invites
        the two to disagree."""
        d = _doc(variants={"k": Config(model="m", name="different")})
        block = d["variants"]["k"]
        assert isinstance(block, str) or "name" not in block

    def test_base_uses_the_same_rule(self):
        assert _doc()["base"] == "base-model"


# ── defaults expanded (AA.3) ─────────────────────────────────────────


class TestDefaultsExpanded:
    def test_expanded_variant_writes_every_decode_field(self):
        """Pinning by omission fails when a default changes, and this
        project has moved a numeric default's meaning twice in two minor
        versions."""
        decode = _doc()["variants"]["sampled"]["decode"]
        for f in ("strategy", "temperature", "top_p", "top_k",
                  "num_samples", "max_new_tokens", "seed"):
            assert f in decode, f

    def test_min_valid_fraction_is_written_even_when_defaulted(self):
        from lmdiff._validity import DEFAULT_MIN_VALID_FRACTION
        d = _doc(min_valid_fraction=DEFAULT_MIN_VALID_FRACTION)
        assert d["min_valid_fraction"] == DEFAULT_MIN_VALID_FRACTION

    def test_metrics_is_a_resolved_list_not_the_word_default(self):
        """``default`` names something that moves. AA.3's test is whether
        a reader can tell from the artifact what was used (§3.3)."""
        d = _doc()
        assert isinstance(d["metrics"], list)
        assert d["metrics"] != "default"
        assert "default" not in d["metrics"]


# ── provenance (§4, §10.3) ───────────────────────────────────────────


class TestProvenance:
    def test_carries_the_six_fields(self):
        p = _doc()[PROVENANCE_KEY]
        for f in ("lmdiff", "python", "torch", "transformers", "geo_schema"):
            assert f in p, f

    def test_omits_what_the_georesult_already_holds(self):
        """The criterion is: record what cannot be recovered from the
        GeoResult, and nothing else. Duplication is the failure this
        project has hit three times (L-035)."""
        p = _doc()[PROVENANCE_KEY]
        for f in ("probes_excluded", "domain_status_summary",
                  "duration_s", "devices"):
            assert f not in p, f

    def test_lm_eval_recorded_only_for_lm_eval_probe_specs(self):
        """An ``lm_eval:`` identifier resolves against an *optional*
        dependency whose version determines probe text, splits and
        ordering. Without it the artifact would pin the metric set, the
        seed and every decode parameter while leaving the probes
        themselves unpinned (§3.3)."""
        assert "lm_eval" not in _doc(probes="v01")[PROVENANCE_KEY]
        assert "lm_eval" in _doc(probes="lm_eval:hellaswag")[PROVENANCE_KEY]

    def test_versions_are_plain_strings(self):
        """``torch.__version__`` is a str *subclass*, and
        ``yaml.safe_dump`` represents exact built-in types only — it
        raises on subclasses rather than falling back."""
        p = _doc()[PROVENANCE_KEY]
        for f in ("lmdiff", "python", "torch", "transformers"):
            if p[f] is not None:
                assert type(p[f]) is str, f


# ── emission (§3.4) ──────────────────────────────────────────────────


class TestEmission:
    def test_round_trips_to_the_same_mapping(self):
        """The load-bearing property: the artifact is a valid input
        (AA.2), so annotation must not change what it parses to."""
        d = _doc()
        assert yaml.safe_load(emit_run_config_yaml(d)) == d

    def test_annotates_inherited_seed(self):
        text = emit_run_config_yaml(_doc())
        assert "inherits the family seed: 42" in text

    def test_annotates_task_override_precedence(self):
        text = emit_run_config_yaml(_doc())
        assert "override the top-level max_new_tokens" in text

    def test_unpinned_seed_is_annotated_differently(self):
        text = emit_run_config_yaml(_doc(seed=None))
        assert "unpinned" in text

    def test_self_check_raises_when_annotation_corrupts(self, monkeypatch):
        """Step 4 lives in the emitter, not only in a test, because the
        failure mode is silent — an artifact that looks right and does
        not load. This is the assertion that the guard is wired, not
        merely present."""
        import lmdiff._runconfig as rc

        monkeypatch.setattr(
            rc, "_annotate",
            lambda text, doc: text.replace("seed: 42", "seed: 43", 1),
        )
        with pytest.raises(RunConfigEmitError, match="would not load back"):
            rc.emit_run_config_yaml(_doc())

    def test_emitted_text_ends_with_a_newline(self):
        assert emit_run_config_yaml(_doc()).endswith("\n")


# ── emission wiring (§6, §10.1) ──────────────────────────────────────


def _result_with_config(tmp_path):
    import lmdiff
    from lmdiff._api import _resolve_metrics

    r = lmdiff.load_result(
        "tests/fixtures/calibration_v041_7variant_summary.json"
    )
    r.run_config_yaml = emit_run_config_yaml(
        _doc(metrics=_resolve_metrics("default"))
    )
    return r


class TestSidecarEmission:
    """Two parts, and implementing it as one function gets it wrong
    (§6): the API boundary produces the text, where the Configs are in
    scope; the report writers write the file, where the path is known.
    """

    def test_sidecar_written_beside_markdown(self, tmp_path):
        r = _result_with_config(tmp_path)
        r.to_markdown(str(tmp_path / "report.md"))
        side = tmp_path / "report.runconfig.yaml"
        assert side.exists()
        assert side.read_text(encoding="utf-8") == r.run_config_yaml

    def test_sidecar_written_beside_html(self, tmp_path):
        r = _result_with_config(tmp_path)
        r.to_html(str(tmp_path / "rep.html"))
        assert (tmp_path / "rep.runconfig.yaml").exists()

    def test_no_sidecar_when_result_carries_no_config(self, tmp_path):
        import lmdiff
        r = lmdiff.load_result(
            "tests/fixtures/calibration_v041_7variant_summary.json"
        )
        assert r.run_config_yaml is None
        r.to_markdown(str(tmp_path / "report.md"))
        assert not (tmp_path / "report.runconfig.yaml").exists()

    def test_no_sidecar_when_rendering_to_a_string(self, tmp_path):
        """No out_path means no place to put it, not an error."""
        r = _result_with_config(tmp_path)
        text = r.to_markdown()
        assert isinstance(text, str)
        assert not list(tmp_path.glob("*.runconfig.yaml"))


class TestHtmlEmbed:
    """HTML embeds a copy where markdown gets only a sidecar (§10.1),
    because HTML's purpose is surviving as one file."""

    def test_html_inlines_the_config(self, tmp_path):
        r = _result_with_config(tmp_path)
        r.to_html(str(tmp_path / "r.html"))
        html = (tmp_path / "r.html").read_text(encoding="utf-8")
        assert '<details class="runconfig">' in html
        assert "lmdiff_schema" in html

    def test_markdown_does_not_inline_it(self, tmp_path):
        r = _result_with_config(tmp_path)
        r.to_markdown(str(tmp_path / "r.md"))
        md = (tmp_path / "r.md").read_text(encoding="utf-8")
        assert "lmdiff_schema" not in md

    def test_bundle_note_only_when_a_ref_is_present(self, tmp_path):
        """A ``__ref__`` resolves against the YAML *file*; an embed has
        no such anchor, so the block says so rather than pretending."""
        from lmdiff.report.html import _build_run_config_block

        r = _result_with_config(tmp_path)
        assert "part of a" not in _build_run_config_block(r)

        r.run_config_yaml = (
            "soft_prompts: {__ref__: soft_prompts.npy}\n"
        )
        assert "bundle" in _build_run_config_block(r)

    def test_embed_absent_when_no_config(self, tmp_path):
        from lmdiff.report.html import _build_run_config_block
        import lmdiff
        r = lmdiff.load_result(
            "tests/fixtures/calibration_v041_7variant_summary.json"
        )
        assert _build_run_config_block(r) == ""
