"""Read-only: does Config survive Config -> dict -> YAML -> dict -> Config?

Investigation script for the v0.4.3 run-config design audit. Answers the
question the audit cannot answer by inspection: is the *existing*
``Config.to_dict`` / ``from_dict`` pair adequate as the basis for a YAML
schema, or does YAML impose constraints JSON did not?

Three stages, because they fail differently:

  1. dict round-trip      Config -> to_dict -> from_dict -> Config
  2. YAML round-trip      + yaml.safe_dump / safe_load in the middle
  3. field-by-field diff  what actually differs, not just == False

Run:  mamba run -n lmdiff python docs/internal/v043_roundtrip_check.py
"""
from __future__ import annotations

import pathlib
import sys

# Run from anywhere: the spec module lives under tests/, which is not a
# package on sys.path unless the repo root is.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from dataclasses import fields, is_dataclass
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

from lmdiff import (
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
from lmdiff._config import _values_equal


# ── the configs under test ───────────────────────────────────────────
# Two sources: the ones the 7-variant calibration actually builds (the
# realistic case), and a maximal Config exercising every field including
# the ones nothing in the project currently uses (the coverage case).


def calibration_configs() -> dict[str, Config]:
    """The Config objects the committed 7-variant fixture was produced
    from. Source of truth is the spec module the test and the regen
    script share."""
    from tests.integration._v041_7variant_spec import build_run_kwargs

    kwargs = build_run_kwargs()
    out: dict[str, Config] = {"__base__": Config(model=kwargs["base"])}
    for name, v in kwargs["variants"].items():
        out[name] = v if isinstance(v, Config) else Config(model=v, name=name)
    return out


def maximal_config() -> Config:
    """Every field populated, including the ones no shipped experiment
    uses. This is where the schema's gaps show up."""
    import numpy as np

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


# ── comparison ───────────────────────────────────────────────────────


def field_diffs(a: Config, b: Config) -> list[tuple[str, Any, Any]]:
    """Field-by-field, numpy-aware. `differs_in` uses `!=` which is
    ambiguous for arrays, so it can't be used here."""
    out = []
    for f in fields(a):
        av, bv = getattr(a, f.name), getattr(b, f.name)
        if not _values_equal(av, bv):
            out.append((f.name, av, bv))
    return out


def yaml_safe(obj: Any) -> tuple[bool, str]:
    """Can yaml.safe_dump represent this, and does safe_load return it?"""
    try:
        text = yaml.safe_dump(obj, sort_keys=True, allow_unicode=True)
    except Exception as e:
        return False, f"safe_dump: {type(e).__name__}: {e}"
    try:
        back = yaml.safe_load(text)
    except Exception as e:
        return False, f"safe_load: {type(e).__name__}: {e}"
    return (back == obj), ("identical" if back == obj else "load != dump input")


def check(label: str, cfg: Config) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label}

    # stage 1 — dict round-trip
    try:
        d = cfg.to_dict()
        back = Config.from_dict(d)
        diffs = field_diffs(cfg, back)
        result["dict_roundtrip"] = "identity" if not diffs else f"{len(diffs)} field(s) differ"
        result["dict_diffs"] = [n for n, _, _ in diffs]
    except Exception as e:
        result["dict_roundtrip"] = f"RAISED {type(e).__name__}: {e}"
        result["dict_diffs"] = []
        return result

    # stage 2 — is the dict YAML-representable at all
    ok, why = yaml_safe(d)
    result["yaml_safe"] = "yes" if ok else f"NO — {why}"

    # stage 3 — full YAML round-trip
    if ok:
        try:
            d2 = yaml.safe_load(yaml.safe_dump(d, sort_keys=True))
            back2 = Config.from_dict(d2)
            diffs2 = field_diffs(cfg, back2)
            result["yaml_roundtrip"] = (
                "identity" if not diffs2 else f"{len(diffs2)} field(s) differ"
            )
            result["yaml_diffs"] = [n for n, _, _ in diffs2]
        except Exception as e:
            result["yaml_roundtrip"] = f"RAISED {type(e).__name__}: {e}"
            result["yaml_diffs"] = []
    else:
        result["yaml_roundtrip"] = "skipped"
        result["yaml_diffs"] = []
    return result


def per_field_yaml_report(cfg: Config) -> list[tuple[str, str, str]]:
    """Classify each field of the serialized dict by YAML friendliness.
    This is the table the audit needs."""
    d = cfg.to_dict()
    rows = []
    for name in sorted(d):
        val = d[name]
        if val is None:
            rows.append((name, "None", "clean (omitted or null)"))
            continue
        ok, why = yaml_safe({name: val})
        kind = type(val).__name__
        if not ok:
            rows.append((name, kind, f"NOT YAML-SAFE — {why}"))
            continue
        # awkward = representable but ugly / lossy-looking
        text = yaml.safe_dump({name: val}, sort_keys=True)
        if "__numpy__" in text or "__numpy_dict__" in text:
            rows.append((name, kind, "awkward — numpy wrapper dict, base64-ish payload"))
        elif len(text) > 400:
            rows.append((name, kind, f"awkward — verbose ({len(text)} chars)"))
        else:
            rows.append((name, kind, "clean"))
    return rows


def main() -> int:
    print("=" * 76)
    print("STAGE A — the Configs the 7-variant calibration actually builds")
    print("=" * 76)
    results = []
    for label, cfg in calibration_configs().items():
        r = check(label, cfg)
        results.append(r)
        print(f"\n{label}")
        print(f"  dict round-trip : {r['dict_roundtrip']}")
        print(f"  yaml-safe       : {r['yaml_safe']}")
        print(f"  yaml round-trip : {r['yaml_roundtrip']}")
        if r.get("dict_diffs") or r.get("yaml_diffs"):
            print(f"  differing fields: dict={r['dict_diffs']} yaml={r['yaml_diffs']}")

    print()
    print("=" * 76)
    print("STAGE B — maximal Config (every field, including unused ones)")
    print("=" * 76)
    mx = maximal_config()
    r = check("maximal", mx)
    print(f"  dict round-trip : {r['dict_roundtrip']}")
    print(f"  yaml-safe       : {r['yaml_safe']}")
    print(f"  yaml round-trip : {r['yaml_roundtrip']}")
    if r.get("dict_diffs") or r.get("yaml_diffs"):
        print(f"  differing fields: dict={r['dict_diffs']} yaml={r['yaml_diffs']}")

    print()
    print("=" * 76)
    print("STAGE C — per-field YAML classification (maximal Config)")
    print("=" * 76)
    print(f"{'field':<26}{'type':<14}verdict")
    for name, kind, verdict in per_field_yaml_report(mx):
        print(f"{name:<26}{kind:<14}{verdict}")

    print()
    print("=" * 76)
    print("STAGE D — what a defaulted Config emits (defaults-expanded question)")
    print("=" * 76)
    plain = Config(model="gpt2")
    d = plain.to_dict()
    print(f"Config(model='gpt2').to_dict() has {len(d)} keys; "
          f"{sum(1 for v in d.values() if v is None)} are None")
    print("non-None keys:", sorted(k for k, v in d.items() if v is not None))
    print()
    print("emitted as YAML (defaults expanded, nothing dropped):")
    print(yaml.safe_dump(d, sort_keys=True, allow_unicode=True))

    failures = [
        r for r in results
        if r["dict_roundtrip"] != "identity" or r["yaml_roundtrip"] != "identity"
    ]
    if r["dict_roundtrip"] != "identity" or r["yaml_roundtrip"] != "identity":
        failures.append(r)
    print("=" * 76)
    print(f"SUMMARY: {len(failures)} config(s) failed to round-trip to identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
