"""Config → YAML → Config round-trips to identity. v0.4.3.

AA.4 mandates this in the first commit of the run-config work, and gives
the reason: a YAML schema maintained separately from ``Config`` grows
fields the Python and CLI paths do not have, and then drifts. This
project has hit that twice — ``magnitudes_per_task_normalized`` diverging
from the field (L-035), and drift-bin edges living in both the
``BoundaryNorm`` and the legend literals.

Promoted from ``docs/internal/v043_roundtrip_check.py``, whose stages A
and B this is. Stage A is the realistic case: the Configs the 7-variant
calibration actually builds. Stage B is the coverage case: every field
populated, including the ones no shipped experiment uses, which is where
gaps show up.

**Ordering note.** This suite could not be written before
``_values_equal`` learned to recurse into dataclasses. With that defect
in place the ``steering`` case failed here on a *correct* serializer, and
the natural reading of the failure was "the serializer loses steering" —
which is a fix to the wrong component. See the design audit §7.4.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import fields

import pytest

yaml = pytest.importorskip("yaml")
np = pytest.importorskip("numpy")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from lmdiff import (  # noqa: E402
    AdapterSpec,
    Config,
    DecodeSpec,
    ICLExample,
    KVCacheSpec,
    Message,
    PruneSpec,
    QuantSpec,
    SteeringSpec,
)
from lmdiff._config import _values_equal  # noqa: E402


def _maximal() -> Config:
    """Every field populated, including ones nothing currently uses."""
    return Config(
        model="meta-llama/Llama-2-7b-hf",
        name="maximal",
        adapter=AdapterSpec(
            type="lora", path="/adapters/x", rank=16,
            target_modules=("q_proj", "v_proj"),
        ),
        quantization=QuantSpec(method="int4", bits=4, compute_dtype="bf16"),
        pruning=PruneSpec(type="unstructured", sparsity=0.5, pattern="2:4"),
        system_prompt="You are concise.",
        icl_examples=(
            ICLExample(user="2+2?", assistant="4", metadata=(("src", "manual"),)),
        ),
        context=(
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ),
        soft_prompts=np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),
        kv_cache_compression=KVCacheSpec(method="h2o", keep_ratio=0.25),
        decode=DecodeSpec(strategy="sample", temperature=1.5, top_k=0, seed=42),
        steering=SteeringSpec(
            vectors={"layer_10": np.array([0.5, 0.6], dtype="float32")},
            scale=2.0, application="add", positions="last",
        ),
        tokenizer_id_override="meta-llama/Llama-2-7b-hf",
        capabilities_required=frozenset({"logits", "hidden_states"}),
        training_recipe_summary="sft on x",
    )


def _calibration_configs() -> dict[str, Config]:
    """The Configs the committed 7-variant fixture was produced from.

    Sourced from the spec module the calibration test and the regen
    script share, so this cannot drift from what the fixture describes.
    """
    from tests.integration._v041_7variant_spec import build_run_kwargs

    kw = build_run_kwargs()
    out = {"__base__": Config(model=kw["base"])}
    for name, v in kw["variants"].items():
        out[name] = v if isinstance(v, Config) else Config(model=v, name=name)
    return out


def _diffs(a: Config, b: Config) -> list[str]:
    """Field-by-field, numpy-aware.

    ``Config.differs_in`` uses plain ``!=``, which is ambiguous for
    arrays, so it cannot be used here — the reason ``_values_equal``
    exists at all.
    """
    return [
        f.name for f in fields(a)
        if not _values_equal(getattr(a, f.name), getattr(b, f.name))
    ]


def _roundtrip(cfg: Config) -> Config:
    return Config.from_dict(yaml.safe_load(yaml.safe_dump(cfg.to_dict())))


# ── stage A: the Configs a real run builds ───────────────────────────


@pytest.mark.parametrize("name", sorted(_calibration_configs()))
def test_calibration_config_round_trips_to_identity(name):
    cfg = _calibration_configs()[name]
    assert _diffs(cfg, _roundtrip(cfg)) == []


# ── stage B: every field, including the unused ones ──────────────────


def test_maximal_config_round_trips_to_identity():
    """Where gaps show up. This is the case that surfaced the
    ``_values_equal`` dataclass defect during the design audit — the
    serializer was correct and the comparator was not."""
    cfg = _maximal()
    assert _diffs(cfg, _roundtrip(cfg)) == []


def test_maximal_config_is_actually_maximal():
    """Guard the guard: if a field is added to ``Config`` and not to the
    fixture, the round-trip test above silently stops covering it."""
    cfg = _maximal()

    def _is_unset(v) -> bool:
        # `v in (None, (), frozenset())` raises on a numpy array — the
        # ambiguous-truth trap `_values_equal` exists to avoid, met here
        # in the test that guards it.
        if v is None:
            return True
        if isinstance(v, (tuple, frozenset, list, dict)):
            return len(v) == 0
        return False

    unset = [f.name for f in fields(cfg) if _is_unset(getattr(cfg, f.name))]
    assert unset == [], (
        f"_maximal() leaves {unset} unpopulated — the round-trip test "
        f"does not cover those fields"
    )


def test_every_config_field_survives_yaml(  # noqa: D103
):
    """Per-field, so a failure names the field rather than the Config."""
    cfg = _maximal()
    back = _roundtrip(cfg)
    for f in fields(cfg):
        assert _values_equal(getattr(cfg, f.name), getattr(back, f.name)), (
            f"{f.name} did not survive Config -> YAML -> Config"
        )


# ── the dict layer, separately from the YAML layer ───────────────────


def test_dict_round_trip_is_identity_too():
    """YAML and ``to_dict`` are separable failure modes: a field can
    survive ``to_dict``/``from_dict`` and still be unrepresentable in
    YAML, or vice versa. Pinning them separately says which broke."""
    cfg = _maximal()
    assert _diffs(cfg, Config.from_dict(cfg.to_dict())) == []


def test_serialized_dict_is_yaml_representable():
    """``yaml.safe_dump`` represents exact built-in types only. A field
    holding a str *subclass* — as ``torch.__version__`` does — raises
    rather than falling back."""
    yaml.safe_dump(_maximal().to_dict())
