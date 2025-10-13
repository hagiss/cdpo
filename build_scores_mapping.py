#!/usr/bin/env python3
"""
Build a lightweight key->scores mapping from annotation shards.

This creates a JSON file mapping sample keys to their 4 scores (pickscore, aesthetic, clip_score, hps_score).
The mapping can be loaded during training to augment the base pickapic_v2 dataset without duplicating images.

Usage:
    CUDA_VISIBLE_DEVICES=1 python build_scores_mapping.py --annotations_dir /data3/jiho/pickapic_annotations_all_scores --output scores_mapping_fix.pkl
"""

import argparse
import json
import os
import pickle
from typing import Dict

import numpy as np
import webdataset as wds
from tqdm import tqdm


def npy_bytes_to_array(npy_bytes: bytes) -> np.ndarray:
    """Convert .npy bytes to numpy array."""
    import io
    return np.load(io.BytesIO(npy_bytes))


def build_scores_mapping(annotations_dir: str) -> Dict[str, Dict[str, list]]:
    """
    Build mapping from sample keys to scores.
    
    Returns:
        Dict mapping key -> {"pickscore": [float, float], "aesthetic": [...], ...}
    """
    shard_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith(".tar")])
    
    if not shard_files:
        raise ValueError(f"No .tar files found in {annotations_dir}")
    
    print(f"Found {len(shard_files)} shard files")
    
    scores_mapping = {}
    
    for shard_file in tqdm(shard_files, desc="Processing annotation shards"):
        shard_path = os.path.join(annotations_dir, shard_file)
        
        # Extract shard identifier from filename (e.g., "000001" from "000001.tar")
        shard_id = os.path.splitext(shard_file)[0]
        
        # Load annotation shard
        ann_ds = wds.WebDataset(
            shard_path,
            handler=wds.ignore_and_continue,
            empty_check=False,
            shardshuffle=False
        )
        
        for sample in ann_ds:
            if not isinstance(sample, dict) or "__key__" not in sample:
                continue
            
            # Create globally unique key: shard_id + local __key__
            local_key = sample["__key__"]
            unique_key = f"{shard_id}_{local_key}"
            # print(unique_key)
            
            # Also store with just local key for backward compatibility (will have collisions)
            # But the unique_key is what should be used
            
            # Extract scores - convert to lists for JSON serialization
            pickscore = npy_bytes_to_array(sample["pickscore.npy"]).tolist() if "pickscore.npy" in sample else [float('nan'), float('nan')]
            aesthetic = npy_bytes_to_array(sample["aesthetic.npy"]).tolist() if "aesthetic.npy" in sample else [float('nan'), float('nan')]
            clip_score = npy_bytes_to_array(sample["clip_score.npy"]).tolist() if "clip_score.npy" in sample else [float('nan'), float('nan')]
            hps_score = npy_bytes_to_array(sample["hps_score.npy"]).tolist() if "hps_score.npy" in sample else [float('nan'), float('nan')]
            
            scores_mapping[unique_key] = {
                "pickscore": pickscore,
                "aesthetic": aesthetic,
                "clip_score": clip_score,
                "hps_score": hps_score,
            }
    
    return scores_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Build key->scores mapping from annotation shards"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        required=True,
        help="Directory containing annotation .tar shards"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scores_mapping.json",
        help="Output file path (supports .json or .pkl)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "pickle"],
        default="pickle",
        help="Output format (auto-detected from extension if not specified)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect format from extension
    if args.format is None:
        if args.output.endswith(".pkl") or args.output.endswith(".pickle"):
            args.format = "pickle"
        else:
            args.format = "json"
    
    print(f"Building scores mapping from {args.annotations_dir}")
    scores_mapping = build_scores_mapping(args.annotations_dir)
    
    print(f"\nBuilt mapping for {len(scores_mapping)} samples")
    
    # Show example
    if scores_mapping:
        example_key = next(iter(scores_mapping))
        print(f"\nExample entry:")
        print(f"  Key: {example_key}")
        print(f"  Scores: {scores_mapping[example_key]}")
    
    # Save mapping
    print(f"\nSaving to {args.output} (format: {args.format})")
    
    if args.format == "json":
        with open(args.output, 'w') as f:
            json.dump(scores_mapping, f, indent=2)
        
        # Print file size
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Saved JSON file: {size_mb:.2f} MB")
    
    elif args.format == "pickle":
        with open(args.output, 'wb') as f:
            pickle.dump(scores_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Saved pickle file: {size_mb:.2f} MB")
    
    print("\n✓ Successfully created scores mapping")
    print(f"\nTo use in training, add:")
    print(f"  --scores_mapping_file {args.output}")


if __name__ == "__main__":
    main()
