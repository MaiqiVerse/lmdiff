"""GPT-2 vs DistilGPT-2: full ModelDiff workflow.

Demonstrates:
1. Loading v01 probe set (90 probes, 3 domains)
2. Metric-level comparison (BD, token entropy, token KL)
3. Per-domain capability radar (accuracy + BD)
4. Terminal report output

Run: mamba run -n modeldiff python examples/gpt2_vs_distilgpt2.py
"""
import warnings

warnings.filterwarnings("ignore")

from modeldiff import Config, ModelDiff, ProbeSet
from modeldiff.report.terminal import print_report, print_radar

# Load probes
probes = ProbeSet.from_json("modeldiff/probes/v01.json")
print(f"Loaded {len(probes)} probes across domains: {probes.domains}")

# Set up comparison
md = ModelDiff(
    Config(model="gpt2"),
    Config(model="distilgpt2"),
    probes,
)

# 1. Metric-level comparison
print("\n=== Running metric comparison ===")
report = md.run(level="output", max_new_tokens=16)
print_report(report)

# 2. Capability radar
print("\n=== Running capability radar ===")
radar_result = md.run_radar(probes=probes, max_new_tokens=16)
print_radar(radar_result)

# 3. JSON export
from modeldiff.report.json_report import write_json

write_json(report, "examples/output_report.json")
write_json(radar_result, "examples/radar_report.json")
print("JSON reports written to examples/output_report.json and examples/radar_report.json")
