#!/usr/bin/env python3
"""
Compute global min/max for score fields in a HuggingFace dataset.

Usage:
    python compute_score_ranges.py \
        --repo_id your-username/your-dataset \
        --split_name train \
        --token hf_xxx  # optional, if private
"""

import argparse
import math
from typing import Dict, Iterable
from tqdm import tqdm
import numpy as np
from datasets import load_dataset


def _update_min_max(current_min: float, current_max: float, values: Iterable[float]):
    arr = np.asarray(list(values), dtype=np.float64).ravel()
    if arr.size == 0:
        return current_min, current_max
    # Filter to finite numbers (drops NaN/Inf)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return current_min, current_max
    new_min = float(valid.min()) if current_min == math.inf else float(min(current_min, valid.min()))
    new_max = float(valid.max()) if current_max == -math.inf else float(max(current_max, valid.max()))
    return new_min, new_max


def compute_score_ranges(ds) -> Dict[str, Dict[str, float]]:
    metrics = ["pickscore", "aesthetic", "clip_score", "hps_score"]
    stats: Dict[str, Dict[str, float]] = {
        m: {"min": math.inf, "max": -math.inf} for m in metrics
    }

    for sample in tqdm(ds):
        for m in metrics:
            values = sample.get(m)
            if values is None:
                continue
            # Expect sequences of two floats; handle any iterable of numbers
            stats[m]["min"], stats[m]["max"] = _update_min_max(
                stats[m]["min"], stats[m]["max"], values
            )

    # Replace infinities with None if nothing was found
    for m in metrics:
        if stats[m]["min"] is math.inf:
            stats[m]["min"] = None
        if stats[m]["max"] is -math.inf:
            stats[m]["max"] = None

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute min/max for score fields in an HF dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="HF dataset repo id, e.g. user/dataset")
    parser.add_argument("--split_name", type=str, default="train", help="Dataset split to analyze (default: train)")
    parser.add_argument("--token", type=str, default=None, help="HF token if the dataset is private")
    args = parser.parse_args()

    ds = load_dataset(args.repo_id, split=args.split_name, token=args.token, streaming=True)
    stats = compute_score_ranges(ds)

    print(f"Dataset: {args.repo_id} [{args.split_name}]")
    for metric, mm in stats.items():
        print(f"- {metric}: min={mm['min']}, max={mm['max']}")


if __name__ == "__main__":
    main()


