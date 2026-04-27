"""GPT-2 vs DistilGPT-2: full lmdiff workflow (v0.3.0 API).

Demonstrates:

1. The bundled v01 probe set (90 probes, 3 domains).
2. Pairwise comparison via :func:`lmdiff.compare`.
3. Capability radar via the existing :class:`ChangeGeometry` flow.
4. JSON export of both results.

Run::

    mamba run -n lmdiff python examples/gpt2_vs_distilgpt2.py
"""
import warnings

warnings.filterwarnings("ignore")

import lmdiff
from lmdiff import Config, ProbeSet
from lmdiff.report.terminal import print_geometry, print_radar

# 1. Load probes
probes = ProbeSet.from_json("lmdiff/probes/v01.json")
print(f"Loaded {len(probes)} probes across domains: {probes.domains}")

# 2. Pairwise comparison via the v0.3.0 top-level API.
print("\n=== Running pairwise comparison ===")
result = lmdiff.compare(
    Config(model="gpt2"),
    Config(model="distilgpt2"),
    probes=probes,
    n_probes=len(probes),
    max_new_tokens=16,
)
print_geometry(result)

# 3. Capability radar (still routed through the v0.2.x ModelDiff helper
#    until the v0.3.0 radar entry point lands in a later commit).
print("\n=== Running capability radar ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from lmdiff import ModelDiff
    md = ModelDiff(Config(model="gpt2"), Config(model="distilgpt2"), probes)
    radar_result = md.run_radar(probes=probes, max_new_tokens=16)
print_radar(radar_result)

# 4. JSON export
from lmdiff.report.json_report import write_json

write_json(result, "examples/output_geometry.json")
write_json(radar_result, "examples/radar_report.json")
print(
    "JSON reports written to examples/output_geometry.json and "
    "examples/radar_report.json"
)
