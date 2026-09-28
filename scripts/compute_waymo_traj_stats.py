#!/usr/bin/env python3
"""Exact per-axis q1/q99 over every native future XY point in Waymo training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


def compute_stats(dataset_jsonl: str | Path) -> dict:
    # Kept as given (repository-relative by convention) so the recorded provenance
    # stays portable instead of embedding the machine's absolute checkout path.
    source = Path(dataset_jsonl).expanduser()
    try:
        import msgspec
    except ImportError:
        decode = json.loads
    else:
        decode = msgspec.json.decode
    digest = hashlib.sha256()
    chunks = []
    chunk_size = 8192
    buffer = np.empty((chunk_size, 20, 2), dtype=np.float64)
    used, count = 0, 0
    with source.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            if not line.strip():
                continue
            row = decode(line)
            if row.get("split", "training") != "training":
                raise ValueError(f"{source}:{line_number}: normalization statistics must use training only")
            # Only future GT participates: not ego history, val, or scored references.
            xy = np.asarray(row["future_traj_4hz"], dtype=np.float64)
            if xy.shape != (20, 2) or not np.isfinite(xy).all():
                raise ValueError(f"{source}:{line_number}: expected 20 finite native XY points")
            if row.get("trajectory_hz", 4) != 4 or row.get("trajectory_horizon_s", 5) != 5:
                raise ValueError(f"{source}:{line_number}: expected native 5s/4Hz GT")
            if "future_valid_mask_4hz" in row and row["future_valid_mask_4hz"] != [1] * 20:
                raise ValueError(f"{source}:{line_number}: GT contains padded/invalid points")
            buffer[used] = xy
            used += 1
            count += 1
            if used == chunk_size:
                chunks.append(buffer)
                buffer = np.empty_like(buffer)
                used = 0
            if count % 50000 == 0:
                print(f"Read {count:,} training samples ({count * 20:,} future XY points)", file=sys.stderr)
    if not count:
        raise ValueError("Training JSONL is empty")
    if used:
        chunks.append(buffer[:used].copy())
    points = np.concatenate(chunks, axis=0).reshape(-1, 2)
    del chunks, buffer
    quantiles = np.percentile(points, [1.0, 99.0], axis=0, method="linear")
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    mean, std = points.mean(axis=0), points.std(axis=0)
    if np.any(quantiles[1] <= quantiles[0]):
        raise ValueError("q99 must exceed q1 for both XY channels")
    axes = {}
    for axis, name in enumerate(("x", "y")):
        axes[name] = {
            "q1": float(quantiles[0, axis]), "q99": float(quantiles[1, axis]),
            "min": float(minimum[axis]), "max": float(maximum[axis]),
            "mean": float(mean[axis]), "std": float(std[axis]),
        }
    return {
        "source": str(source), "source_sha256": digest.hexdigest(), "split": "training",
        "num_samples": count, "num_points": count * 20,
        "trajectory_hz": 4, "trajectory_horizon_s": 5, "action_dim": 2,
        "quantile_low_pct": 1.0, "quantile_high_pct": 99.0,
        "normalization": "2 * (value - q1) / (q99 - q1) - 1; no clipping",
        "future_traj_4hz": axes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", default="./data/waymo_training_front_video4s_traj5s_4hz_xy.jsonl")
    parser.add_argument("--output", default="./data/waymo_dataset_stats.json")
    args = parser.parse_args()
    result = compute_stats(args.dataset_jsonl)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False, encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary = Path(handle.name)
    temporary.replace(output)
    print(f"Saved {output}: {result['num_samples']:,} samples, {result['num_points']:,} points")
    for name, values in result["future_traj_4hz"].items():
        print(f"{name}: normalization min(q1)={values['q1']:.12g}, max(q99)={values['q99']:.12g}")


if __name__ == "__main__":
    main()
