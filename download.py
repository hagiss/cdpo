#!/usr/bin/env python
"""
Download WebDataset tar files from HuggingFace Hub to local directory.
This allows you to download once and then use the local files as an iterable dataset.

Usage:
    python download_webdataset.py --dataset_name USER/DATASET --output_dir ./data/webdataset
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_webdataset(dataset_name, output_dir, config_name=None, split="train"):
    """
    Download webdataset tar files from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "username/dataset")
        output_dir: Local directory to save tar files
        config_name: Optional dataset config/subset name
        split: Dataset split to download (default: "train")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {dataset_name} to {output_dir}")
    
    # Load dataset without streaming to download all files
    print("Loading dataset metadata...")
    dataset = load_dataset(
        dataset_name,
        config_name,
        split=split,
        cache_dir=None,  # Use default HF cache
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # For webdataset format, the actual tar files are in the HF cache
    # We need to use a different approach - download with streaming first to get cache,
    # then copy the files
    
    # Alternative: Use huggingface_hub to download files directly
    from huggingface_hub import snapshot_download
    
    print(f"Downloading dataset files to {output_dir}...")
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns="*.tar",  # Only download tar files
        ignore_patterns=[".*", "*.md", "*.txt", "*.json"],  # Skip metadata files if not needed
    )
    
    print(f"✓ Download complete! WebDataset tar files saved to {output_dir}")
    
    # List downloaded files
    tar_files = sorted(output_path.glob("**/*.tar"))
    print(f"\nDownloaded {len(tar_files)} tar files:")
    for tar_file in tar_files[:10]:  # Show first 10
        print(f"  - {tar_file.relative_to(output_path)}")
    if len(tar_files) > 10:
        print(f"  ... and {len(tar_files) - 10} more")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Download WebDataset from HuggingFace Hub")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., 'username/dataset')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Local directory to save tar files",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset config/subset name (optional)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (default: train)",
    )
    
    args = parser.parse_args()
    
    output_dir = download_webdataset(
        args.dataset_name,
        args.output_dir,
        args.dataset_config_name,
        args.split,
    )
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

