#!/usr/bin/env python3
"""
Build HuggingFace Dataset from annotations created by annotate_with_all_scores.py

This converts WebDataset shards (.tar files) into a HuggingFace Dataset format
that can be easily loaded with datasets.load_dataset().

Usage:
    python build_all_scores_hf_dataset.py \
        --annotations_dir /data3/jiho/pickapic_annotations_all_scores \
        --repo_id hagiss/mvv_full_all_scores \
        --tokens 
"""

import argparse
import io
import os
from typing import Dict, Iterator

import numpy as np
import webdataset as wds
from PIL import Image
from datasets import Dataset, Features, Value, Sequence, Image as HFImage
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm


def pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to JPEG bytes."""
    with io.BytesIO() as buf:
        image.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def npy_bytes_to_array(npy_bytes: bytes) -> np.ndarray:
    """Convert .npy bytes to numpy array."""
    return np.load(io.BytesIO(npy_bytes))


def _is_image_key(k: str) -> bool:
    """Check if a key represents an image in WebDataset."""
    if not isinstance(k, str):
        return False
    lower = k.lower()
    if lower.endswith((".jpg", ".jpeg", ".png")):
        return True
    base = lower.split(";")[0]
    if base in ("jpg", "jpeg", "png"):
        return True
    if base.startswith("jpg_") or base.startswith("jpeg_") or base.startswith("png_"):
        return True
    return False


def iter_samples_from_local_shards(
    annotations_dir: str,
    include_images: bool = False,
) -> Iterator[Dict]:
    """
    Iterate over samples from local annotation shards.
    
    Args:
        annotations_dir: Directory containing .tar shard files
        include_images: If True, include image bytes (much larger dataset)
    """
    shard_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith(".tar")])
    
    if not shard_files:
        raise ValueError(f"No .tar files found in {annotations_dir}")
    
    print(f"Found {len(shard_files)} shard files")
    
    for shard_file in tqdm(shard_files, desc="Processing shards"):
        shard_path = os.path.join(annotations_dir, shard_file)
        
        # Load source dataset to get images and prompts
        # Note: This assumes you still have access to the original dataset
        # If not, you'll need to include images in the annotation shards
        src_shard_name = shard_file
        try:
            src_local = hf_hub_download(
                repo_id="sayakpaul/pickapic_v2_webdataset",
                filename=src_shard_name,
                repo_type="dataset"
            )
            src_ds = wds.WebDataset(
                src_local,
                handler=wds.ignore_and_continue,
                empty_check=False,
                shardshuffle=False
            ).decode("pil")
            
            # Create lookup for source data
            src_data = {}
            for sample in src_ds:
                if isinstance(sample, dict) and "__key__" in sample:
                    key = sample["__key__"]
                    cand = sorted([k for k in sample.keys() if _is_image_key(k)])
                    src_data[key] = {
                        "img0": sample.get(cand[0]) if len(cand) > 0 else None,
                        "img1": sample.get(cand[1]) if len(cand) > 1 else None,
                        "prompt": sample.get("original_prompt.txt", b"").decode("utf-8") if isinstance(sample.get("original_prompt.txt"), bytes) else "",
                        "label": sample.get("label_0", -1),
                    }
        except Exception as e:
            print(f"Warning: Could not load source data for {shard_file}: {e}")
            src_data = {}
        
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
            
            key = sample["__key__"]
            
            # Get source data if available
            src = src_data.get(key, {})
            
            # Extract scores
            pickscore = npy_bytes_to_array(sample["pickscore.npy"]) if "pickscore.npy" in sample else np.array([np.nan, np.nan])
            aesthetic = npy_bytes_to_array(sample["aesthetic.npy"]) if "aesthetic.npy" in sample else np.array([np.nan, np.nan])
            clip_score = npy_bytes_to_array(sample["clip_score.npy"]) if "clip_score.npy" in sample else np.array([np.nan, np.nan])
            hps_score = npy_bytes_to_array(sample["hps_score.npy"]) if "hps_score.npy" in sample else np.array([np.nan, np.nan])
            
            result = {
                "key": key,
                "caption": src.get("prompt", ""),
                "label_0": src.get("label", -1),
                "pickscore": pickscore.tolist(),
                "aesthetic": aesthetic.tolist(),
                "clip_score": clip_score.tolist(),
                "hps_score": hps_score.tolist(),
            }
            
            # Optionally include images
            if include_images:
                img0 = src.get("img0")
                img1 = src.get("img1")
                if img0 and img1:
                    result["jpg_0"] = pil_to_jpeg_bytes(img0)
                    result["jpg_1"] = pil_to_jpeg_bytes(img1)
            
            yield result


def main():
    parser = argparse.ArgumentParser(
        description="Build HuggingFace Dataset from all-scores annotations"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="./pickapic_annotations_all_scores",
        help="Directory containing annotation .tar shards"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target HF dataset repo, e.g., your-username/pickapic-all-scores"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private dataset"
    )
    parser.add_argument(
        "--include_images",
        action="store_true",
        help="Include image bytes (warning: much larger dataset!)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for authentication"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Dataset split name (default: train)"
    )
    
    args = parser.parse_args()
    
    # Define dataset schema
    base_features = {
        "key": Value("string"),
        "caption": Value("string"),
        "label_0": Value("int32"),
        "pickscore": Sequence(Value("float32"), length=2),
        "aesthetic": Sequence(Value("float32"), length=2),
        "clip_score": Sequence(Value("float32"), length=2),
        "hps_score": Sequence(Value("float32"), length=2),
    }
    
    if args.include_images:
        base_features["jpg_0"] = Value("binary")
        base_features["jpg_1"] = Value("binary")
    
    features = Features(base_features)
    
    print(f"Building dataset from {args.annotations_dir}")
    print(f"Include images: {args.include_images}")
    print()
    
    # Create dataset from generator
    ds = Dataset.from_generator(
        lambda: iter_samples_from_local_shards(
            args.annotations_dir,
            include_images=args.include_images
        ),
        features=features,
    )
    
    print(f"\nDataset created: {ds}")
    print(f"Number of samples: {len(ds)}")
    print(f"Features: {list(ds.features.keys())}")
    print()
    
    # Push to hub
    print(f"Uploading to https://huggingface.co/datasets/{args.repo_id}")
    ds.push_to_hub(
        args.repo_id,
        private=args.private,
        token=args.token,
        split=args.split_name,
    )
    
    print(f"\n✓ Successfully uploaded to https://huggingface.co/datasets/{args.repo_id}")
    print("\nTo load this dataset:")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('{args.repo_id}')")
    print(f"  print(ds['train'][0])  # First sample")


if __name__ == "__main__":
    main()

