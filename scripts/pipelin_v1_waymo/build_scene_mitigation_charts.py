#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build scene-level mitigation counterfactual charts for Waymo audit outputs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-waymo-audit")

from common import DEFAULT_DATAROOT, mkdir, output_root


def int_value(value: str) -> int:
    return int(float(value))


def scene_sample_rows(sample_csv: Path) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "scene": "",
            "samples": 0,
            "improved": 0,
            "unchanged": 0,
            "worse": 0,
        }
    )
    with sample_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scene = row["scene"]
            item = grouped[scene]
            item["scene"] = scene
            item["samples"] = int(item["samples"]) + 1
            delta = int_value(row["delta_score"])
            if delta > 0:
                item["improved"] = int(item["improved"]) + 1
            elif delta < 0:
                item["worse"] = int(item["worse"]) + 1
            else:
                item["unchanged"] = int(item["unchanged"]) + 1
    rows = list(grouped.values())
    rows.sort(key=lambda row: (-(int(row["improved"]) + int(row["worse"])), str(row["scene"])))
    return rows


def write_scene_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    mkdir(path.parent)
    fields = [
        "scene",
        "samples",
        "improved",
        "unchanged",
        "worse",
        "improved_rate",
        "worse_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            samples = int(row["samples"])
            out = {
                **row,
                "improved_rate": round(int(row["improved"]) / samples, 4) if samples else 0.0,
                "worse_rate": round(int(row["worse"]) / samples, 4) if samples else 0.0,
            }
            writer.writerow(out)


def plot_sample_change_stacked(rows: List[Dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = rows[:20][::-1]
    labels = [str(row["scene"]) for row in selected]
    improved = [int(row["improved"]) for row in selected]
    unchanged = [int(row["unchanged"]) for row in selected]
    worse = [int(row["worse"]) for row in selected]

    height = max(4.8, 0.65 * len(selected) + 1.8)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(labels, improved, label="Safer", color="#54A24B")
    ax.barh(labels, unchanged, left=improved, label="Unchanged", color="#BAB0AC")
    left_for_worse = [a + b for a, b in zip(improved, unchanged)]
    ax.barh(labels, worse, left=left_for_worse, label="Riskier", color="#E45756")
    ax.set_title("Scene sample-level change composition", fontsize=14, pad=12)
    ax.set_xlabel("Target-agent samples")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    totals = [a + b + c for a, b, c in zip(improved, unchanged, worse)]
    offset = max(max(totals) * 0.01 if totals else 0, 1.0)
    ax.set_xlim(right=(max(totals) if totals else 0) + offset * 8)
    for index, total in enumerate(totals):
        ax.text(total + offset, index, str(total), va="center", fontsize=9)

    fig.tight_layout()
    mkdir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None, help="Default: <dataroot>/nurisk_style")
    parser.add_argument(
        "--audit-dir",
        default=None,
        help="Default: <input-dir>/mitigation_counterfactual_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    risk_root = output_root(str(dataroot), args.input_dir)
    audit_dir = Path(args.audit_dir).expanduser().resolve() if args.audit_dir else risk_root / "mitigation_counterfactual_audit"
    sample_csv = audit_dir / "mitigation_counterfactual_comprehensive_samples.csv"
    if not sample_csv.exists():
        raise FileNotFoundError(f"Missing sample CSV: {sample_csv}")

    rows = scene_sample_rows(sample_csv)
    scene_csv = audit_dir / "mitigation_counterfactual_by_scene_samples.csv"
    chart_path = audit_dir / "charts" / "18_scene_sample_change_stacked_top20.png"
    write_scene_csv(scene_csv, rows)
    plot_sample_change_stacked(rows, chart_path)

    print("Waymo scene-level mitigation chart done.")
    print(f"Scenes: {len(rows)}")
    print(f"Scene CSV: {scene_csv}")
    print(f"Chart: {chart_path}")


if __name__ == "__main__":
    main()
