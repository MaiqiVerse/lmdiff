#!/usr/bin/env python
"""Plot all geometry figures from a GeoResult JSON.

Reads a v1/v2/v3 GeoResult JSON (produced by ChangeGeometry.analyze() +
lmdiff.report.json_report.to_json) and emits:
  - direction_heatmap.png  (cosine similarity matrix)
  - selective_heatmap.png  (Pearson-r matrix, v2+)
  - pca_scatter.png        (variants projected onto PC1-PC2)
  - domain_bar.png         (per-variant per-domain δ magnitude, v3 only)
  - index.html             (static preview indexing the above)

Plots that require data not present (e.g. probe_domains for domain_bar,
selective_cosine_matrix for selective_heatmap, n_variants >= 2 for PCA)
are skipped with a printed warning; the rest still produce.

Usage:
    mamba run -n lmdiff python scripts/plot_family_geometry.py \\
        examples/family_geometry_extended.json \\
        --output-dir examples/family_geometry_extended_figures/

Requires: pip install lmdiff-kit[viz]
"""
from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

from lmdiff.geometry import GeoResult
from lmdiff.report.json_report import geo_result_from_json_dict


def _write_index_html(output_dir: Path, entries: list[tuple[str, str]]) -> Path:
    """entries = [(title, filename)]. Writes index.html next to the figures."""
    rows = []
    for title, filename in entries:
        rows.append(
            f'<figure><figcaption>{escape(title)}</figcaption>'
            f'<img src="{escape(filename)}" alt="{escape(title)}" style="max-width:100%;"/>'
            f'</figure>'
        )
    html = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        '<title>lmdiff geometry figures</title>'
        '<style>body{font-family:system-ui,sans-serif;margin:2em;max-width:900px}'
        'figure{margin:2em 0;padding:1em;border:1px solid #ddd;border-radius:8px}'
        'figcaption{font-weight:600;margin-bottom:0.5em;color:#333}'
        '</style></head><body>'
        '<h1>lmdiff — GeoResult figures</h1>'
        + "\n".join(rows)
        + "</body></html>"
    )
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _plot_or_warn(label: str, func, **kwargs) -> str | None:
    """Call `func(**kwargs)`; on any exception print a warning and return None."""
    try:
        out = func(**kwargs)
        print(f"  {label}: {out}")
        return out
    except Exception as exc:  # noqa: BLE001 - per-plot isolation
        print(f"  [WARN] {label} skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot all geometry figures from a GeoResult JSON.")
    parser.add_argument("input", type=Path, help="GeoResult JSON path")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write PNGs + index.html (created if missing)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: {args.input} does not exist", file=sys.stderr)
        return 2

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)
    geo: GeoResult = geo_result_from_json_dict(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded GeoResult: {len(geo.variant_names)} variants × {geo.n_probes} probes")
    print(f"Writing to {args.output_dir}")

    entries: list[tuple[str, str]] = []

    # 1. Direction heatmap (always available from v1+)
    from lmdiff.viz.direction_heatmap import plot_direction_heatmap
    out = _plot_or_warn(
        "direction_heatmap",
        plot_direction_heatmap,
        cosine_matrix=geo.cosine_matrix,
        variant_names=geo.variant_names,
        out_path=args.output_dir / "direction_heatmap.png",
        title=f"Direction similarity: {geo.base_name} vs {len(geo.variant_names)} variants",
    )
    if out is not None:
        entries.append(("Direction similarity (cosine)", "direction_heatmap.png"))

    # 2. Selective heatmap (v2+ only)
    if geo.selective_cosine_matrix:
        out = _plot_or_warn(
            "selective_heatmap",
            plot_direction_heatmap,
            cosine_matrix=geo.selective_cosine_matrix,
            variant_names=geo.variant_names,
            out_path=args.output_dir / "selective_heatmap.png",
            title="Selective cosine (Pearson r, mean-removed)",
        )
        if out is not None:
            entries.append(("Selective cosine (Pearson r)", "selective_heatmap.png"))
    else:
        print("  [WARN] selective_heatmap skipped: selective_cosine_matrix empty "
              "(legacy v1 JSON)", file=sys.stderr)

    # 3. PCA scatter (needs n_variants >= 2)
    if len(geo.variant_names) < 2:
        print("  [WARN] pca_scatter skipped: n_variants < 2", file=sys.stderr)
    else:
        try:
            pca_result = geo.pca_map(n_components=2)
        except ValueError as exc:
            print(f"  [WARN] pca_scatter skipped: pca_map failed: {exc}", file=sys.stderr)
        else:
            from lmdiff.viz.pca_scatter import plot_pca_scatter
            out = _plot_or_warn(
                "pca_scatter",
                plot_pca_scatter,
                pca_result=pca_result,
                out_path=args.output_dir / "pca_scatter.png",
                title=f"Change geometry (PCA) — base: {geo.base_name}",
            )
            if out is not None:
                entries.append(("PCA scatter", "pca_scatter.png"))

    # 4. Domain bar (needs probe_domains — v3)
    if not geo.probe_domains:
        print("  [WARN] domain_bar skipped: probe_domains empty "
              "(legacy v1/v2 JSON or list[str] prompts)", file=sys.stderr)
    else:
        try:
            heatmap = geo.domain_heatmap()
        except ValueError as exc:
            print(f"  [WARN] domain_bar skipped: domain_heatmap failed: {exc}",
                  file=sys.stderr)
        else:
            from lmdiff.viz.domain_bar import plot_domain_bar
            out = _plot_or_warn(
                "domain_bar",
                plot_domain_bar,
                domain_heatmap=heatmap,
                out_path=args.output_dir / "domain_bar.png",
                title=f"Per-domain δ magnitude — base: {geo.base_name}",
            )
            if out is not None:
                entries.append(("Per-domain δ magnitude", "domain_bar.png"))

    # Index
    if entries:
        idx = _write_index_html(args.output_dir, entries)
        print(f"\nIndex: {idx}")
        print(f"Figures rendered: {len(entries)}")
    else:
        print("\n[WARN] no figures rendered", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
