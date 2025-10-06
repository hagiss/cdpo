#!/usr/bin/env python3
"""
Multi-GPU version of the batched annotation pipeline with optimizations.

This script automatically distributes shards across available GPUs.

OPTIMIZATIONS APPLIED:
    - True batch inference: All 4 models (PickScore, Aesthetic, CLIP, HPS) now process
      full batches in parallel instead of looping through samples
    - Batch-level prefetching: Complete batches prepared in background (4-batch buffer)
    - Async disk writing: Results written to HDD in background thread (non-blocking)
    - CUDA optimizations: cudnn.benchmark enabled, async transfers, cache management
    - Memory efficiency: Batch processing, cache clearing between shards
    
    Pipeline: [Load Batch] → [GPU Process] → [Queue Write] → [Load Next Batch]
                    ↓ (background)              ↓ (background)
              [Batch Ready]              [Disk Write Thread]
    
    Expected GPU utilization: 95-100% sustained (up from 10-30% intermittent)

Usage:
    # Auto-detect and use all GPUs
    python annotate_with_all_scores_multigpu.py
    
    # Use specific GPUs with larger batch size
    CUDA_VISIBLE_DEVICES=0,1,2 BATCH_SIZE=32 python annotate_with_all_scores_multigpu.py
    
    # Manual shard range (for distributed jobs)
    GPU_ID=0 START_SHARD=0 END_SHARD=80 python annotate_with_all_scores_multigpu.py

Environment Variables:
    REPO_ID: HuggingFace dataset repo
    TARGET_SAMPLES: Number of samples (use "ALL" for full dataset)
    OUT_DIR: Output directory
    BATCH_SIZE: Batch size per GPU (default: 16, recommend 24-48 for high-end GPUs)
    GPU_ID: Manual GPU selection (for distributed mode)
    START_SHARD: Starting shard index (for distributed mode)
    END_SHARD: Ending shard index (for distributed mode)
"""

import io
import os
import math
import json
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from queue import Queue
from threading import Thread, Lock

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.pickscore_utils import Selector as PickScoreSelector
from utils.aes_utils import Selector as AestheticSelector
from utils.clip_utils import Selector as CLIPSelector
from utils.hps_utils import Selector as HPSSelector


# --- Configuration ---

REPO_ID = os.environ.get("REPO_ID", "sayakpaul/pickapic_v2_webdataset")
REPO_TYPE = "dataset"
TARGET_SAMPLES = os.environ.get("TARGET_SAMPLES", "50000")
TARGET_SAMPLES = None if TARGET_SAMPLES in ("0", "all", "ALL", "None") else int(TARGET_SAMPLES)
SAMPLES_PER_SHARD = 1500

OUT_DIR = os.environ.get("OUT_DIR", "/data3/jiho/pickapic_annotations_all_scores")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
FORCE_REWRITE = str(os.environ.get("FORCE_REWRITE", "")).lower() in ("1", "true", "yes")

# Multi-GPU specific
GPU_ID = os.environ.get("GPU_ID", None)  # For manual mode
START_SHARD = int(os.environ.get("START_SHARD", "0"))
END_SHARD = os.environ.get("END_SHARD", None)
END_SHARD = None if END_SHARD is None else int(END_SHARD)


# --- Helper Functions ---

class AsyncTarWriter:
    """Asynchronous WebDataset tar writer that performs disk I/O in background thread."""
    
    def __init__(self, tar_path, queue_size=10):
        self.tar_path = tar_path
        self.queue = Queue(maxsize=queue_size)
        self.writer_thread = None
        self.exception = None
        self._stop = False
    
    def _writer_worker(self):
        """Background thread that writes to disk."""
        try:
            with wds.TarWriter(self.tar_path) as sink:
                while True:
                    item = self.queue.get()
                    if item is None:  # Sentinel to stop
                        break
                    sink.write(item)
        except Exception as e:
            self.exception = e
    
    def start(self):
        """Start the background writer thread."""
        self.writer_thread = Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()
    
    def write(self, sample_dict):
        """Queue a sample for writing (non-blocking for GPU processing)."""
        if self.exception:
            raise self.exception
        self.queue.put(sample_dict)
    
    def close(self):
        """Close the writer and wait for all pending writes to complete."""
        self.queue.put(None)  # Sentinel
        if self.writer_thread:
            self.writer_thread.join()
        if self.exception:
            raise self.exception
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BatchPrefetcher:
    """Prefetch and prepare complete batches in background threads to hide I/O latency.

    Optionally applies CPU-side preprocessing to produce tensors for downstream
    GPU models (CLIP/HPS) so that the GPU step does not wait on CPU transforms.
    """
    
    def __init__(self, dataset_iterator, batch_size, prefetch_batches=4, num_workers=2,
                 preprocess_image_fn=None, pin_memory=True):
        self.dataset_iterator = dataset_iterator
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers
        self.queue = Queue(maxsize=prefetch_batches)
        self.threads = []
        self._stop = False
        self.preprocess_image_fn = preprocess_image_fn
        self.pin_memory = pin_memory and torch.cuda.is_available()
    
    def _parse_sample(self, sample):
        """Parse a single sample from WebDataset."""
        if not isinstance(sample, dict):
            return None
        
        key = sample.get("__key__")
        candidate_keys = sorted([k for k in sample.keys() if _is_image_key(k)])
        img0 = sample.get(candidate_keys[0]) if len(candidate_keys) > 0 else None
        img1 = sample.get(candidate_keys[1]) if len(candidate_keys) > 1 else None
        prompt_bytes = sample.get("original_prompt.txt")
        
        if key is None or not isinstance(img0, Image.Image) or not isinstance(img1, Image.Image):
            return None
        
        # Convert images to RGB immediately (faster in background thread)
        img0 = img0.convert('RGB') if img0.mode != 'RGB' else img0
        img1 = img1.convert('RGB') if img1.mode != 'RGB' else img1
        
        # Optional CPU-side preprocessing for CLIP/HPS
        clip_t0 = None
        clip_t1 = None
        if self.preprocess_image_fn is not None:
            try:
                clip_t0 = self.preprocess_image_fn(img0)
                clip_t1 = self.preprocess_image_fn(img1)
                if self.pin_memory:
                    clip_t0 = clip_t0.pin_memory()
                    clip_t1 = clip_t1.pin_memory()
            except Exception:
                clip_t0, clip_t1 = None, None
        
        # Decode prompt
        prompt = ""
        if isinstance(prompt_bytes, (bytes, bytearray)):
            try:
                prompt = prompt_bytes.decode("utf-8", errors="ignore")
            except Exception:
                prompt = ""
        elif isinstance(prompt_bytes, str):
            prompt = prompt_bytes
        
        return {
            'key': key,
            'img0': img0,  # PIL for PickScore/Aesthetic processors
            'img1': img1,
            'clip_t0': clip_t0,  # torch.Tensor for CLIP/HPS (CPU, optionally pinned)
            'clip_t1': clip_t1,
            'prompt': prompt
        }
    
    def _producer(self):
        """Background thread that loads and batches data."""
        try:
            batch_keys, batch_images_0, batch_images_1, batch_prompts = [], [], [], []
            batch_clip_t0, batch_clip_t1 = [], []
            
            for sample in self.dataset_iterator:
                if self._stop:
                    break
                
                parsed = self._parse_sample(sample)
                if parsed is None:
                    continue
                
                batch_keys.append(parsed['key'])
                batch_images_0.append(parsed['img0'])
                batch_images_1.append(parsed['img1'])
                batch_prompts.append(parsed['prompt'])
                batch_clip_t0.append(parsed['clip_t0'])
                batch_clip_t1.append(parsed['clip_t1'])
                
                # When batch is full, put it in queue
                if len(batch_keys) >= self.batch_size:
                    batch = {
                        'keys': batch_keys,
                        'images_0': batch_images_0,
                        'images_1': batch_images_1,
                        'prompts': batch_prompts,
                        'clip_t0': batch_clip_t0,
                        'clip_t1': batch_clip_t1
                    }
                    self.queue.put(batch)
                    batch_keys, batch_images_0, batch_images_1, batch_prompts = [], [], [], []
                    batch_clip_t0, batch_clip_t1 = [], []
            
            # Put remaining samples as final batch
            if batch_keys:
                batch = {
                    'keys': batch_keys,
                    'images_0': batch_images_0,
                    'images_1': batch_images_1,
                    'prompts': batch_prompts,
                    'clip_t0': batch_clip_t0,
                    'clip_t1': batch_clip_t1
                }
                self.queue.put(batch)
            
            self.queue.put(None)  # Sentinel
        except Exception as e:
            self.queue.put(e)
    
    def __iter__(self):
        self._stop = False
        # Start producer thread
        thread = Thread(target=self._producer, daemon=True)
        thread.start()
        self.threads.append(thread)
        return self
    
    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        if isinstance(batch, Exception):
            raise batch
        return batch
    
    def stop(self):
        """Stop the prefetching threads."""
        self._stop = True
        for thread in self.threads:
            thread.join(timeout=1.0)


def list_hf_shards(repo_id: str, repo_type: str = "dataset"):
    """List all .tar shards in a HuggingFace dataset repository."""
    files = list_repo_files(repo_id, repo_type=repo_type)
    shard_files = sorted([f for f in files if f.endswith(".tar")])
    if not shard_files:
        raise RuntimeError("No .tar shards found in the dataset repository.")
    return shard_files


def get_available_gpus():
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []
    
    # Check CUDA_VISIBLE_DEVICES
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_devices:
        return [int(x) for x in visible_devices.split(",")]
    
    return list(range(torch.cuda.device_count()))


def split_shards(shards: List[str], num_splits: int) -> List[List[str]]:
    """Split shards evenly across GPUs."""
    shard_splits = [[] for _ in range(num_splits)]
    for i, shard in enumerate(shards):
        shard_splits[i % num_splits].append(shard)
    return shard_splits


def array_to_npy_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array to bytes for WebDataset storage."""
    with io.BytesIO() as f:
        np.save(f, arr)
        return f.getvalue()


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


# --- Import Batched Scorer ---
# (Same as in annotate_with_all_scores_batched.py)

class BatchedUnifiedScorer:
    """Batched wrapper that loads all 4 scoring models and provides batch inference."""
    
    def __init__(self, device: str, batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size
        
        # Initialize all models
        self.pickscore = PickScoreSelector(device)
        self.aesthetic = AestheticSelector(device)
        self.clip = CLIPSelector(device)
        self.hps = HPSSelector(device)
    
    def score_batch(self, images_0: List[Image.Image], images_1: List[Image.Image], 
                   prompts: List[str],
                   clip_tensors0: Optional[List[torch.Tensor]] = None,
                   clip_tensors1: Optional[List[torch.Tensor]] = None) -> List[Dict[str, np.ndarray]]:
        """Score a batch of image pairs with all 4 metrics using true batched inference.

        If clip_tensors0/clip_tensors1 are provided (CPU tensors, optionally pinned),
        they will be used for CLIP/HPS image encodings to avoid CPU-side preprocessing here.
        """
        batch_size = len(images_0)
        images_0 = [img.convert('RGB') if img.mode != 'RGB' else img for img in images_0]
        images_1 = [img.convert('RGB') if img.mode != 'RGB' else img for img in images_1]
        
        results = [{"pickscore": None, "aesthetic": None, "clip": None, "hps": None} 
                   for _ in range(batch_size)]
        
        # PickScore - Batched processing (encode text once)
        try:
            # Process all image pairs with their prompts in parallel batches
            all_images = images_0 + images_1  # [img0_0, img0_1, ..., img1_0, img1_1, ...]
            
            # Batch process through PickScore
            image_inputs = self.pickscore.processor(
                images=all_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            
            text_inputs = self.pickscore.processor(
                text=prompts,  # encode each prompt once
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                image_embs = self.pickscore.model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                
                text_embs = self.pickscore.model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                
                # Compute scores per pair without duplicating text encodings
                scores0 = (text_embs * image_embs[:batch_size]).sum(dim=-1).cpu().numpy()
                scores1 = (text_embs * image_embs[batch_size:]).sum(dim=-1).cpu().numpy()
                for i in range(batch_size):
                    results[i]["pickscore"] = np.array([scores0[i], scores1[i]], dtype=np.float32)
        except Exception as e:
            for i in range(batch_size):
                results[i]["pickscore"] = np.array([np.nan, np.nan], dtype=np.float32)
        
        # Aesthetic Score - Already batched correctly
        try:
            all_images_flat = images_0 + images_1
            aesthetic_scores = self.aesthetic.score(all_images_flat, "")
            for i in range(batch_size):
                results[i]["aesthetic"] = np.array([
                    aesthetic_scores[i], aesthetic_scores[batch_size + i]
                ], dtype=np.float32)
        except Exception:
            for i in range(batch_size):
                results[i]["aesthetic"] = np.array([np.nan, np.nan], dtype=np.float32)
        
        # CLIP Score - Batched processing (reuse preprocessed tensors if provided; encode text once)
        try:
            # Prepare image tensors
            if clip_tensors0 is not None and clip_tensors1 is not None and len(clip_tensors0) == batch_size and len(clip_tensors1) == batch_size:
                image_tensors = torch.stack(clip_tensors0 + clip_tensors1)
            else:
                all_images = images_0 + images_1
                image_tensors = torch.stack([self.clip.preprocess_val(img) for img in all_images])
            image_tensors = image_tensors.to(device=self.device, non_blocking=True)

            # Tokenize/encode text once
            text_tokens = self.clip.tokenizer(prompts).to(device=self.device, non_blocking=True)
            
            with torch.no_grad():
                # Encode separately to avoid recomputing text twice
                image_features = self.clip.model.encode_image(image_tensors, normalize=True)
                text_features = self.clip.model.encode_text(text_tokens, normalize=True)
                
                # Split into pairs and score
                scores0 = (image_features[:batch_size] * text_features).sum(dim=-1).cpu().numpy()
                scores1 = (image_features[batch_size:] * text_features).sum(dim=-1).cpu().numpy()
                for i in range(batch_size):
                    results[i]["clip"] = np.array([scores0[i], scores1[i]], dtype=np.float32)
        except Exception as e:
            for i in range(batch_size):
                results[i]["clip"] = np.array([np.nan, np.nan], dtype=np.float32)
        
        # HPS Score - Batched processing (reuse preprocessed tensors if provided; encode text once)
        try:
            # Prepare image tensors
            if clip_tensors0 is not None and clip_tensors1 is not None and len(clip_tensors0) == batch_size and len(clip_tensors1) == batch_size:
                hps_images = torch.stack(clip_tensors0 + clip_tensors1)
            else:
                all_images = images_0 + images_1
                hps_images = torch.stack([self.hps.preprocess_val(img) for img in all_images])
            hps_images = hps_images.to(device=self.device, non_blocking=True)

            # Tokenize/encode text once
            text_tokens = self.hps.tokenizer(prompts).to(device=self.device, non_blocking=True)
            
            with torch.no_grad():
                # Encode separately to avoid recomputing text twice
                image_features = self.hps.model.encode_image(hps_images, normalize=True)
                text_features = self.hps.model.encode_text(text_tokens, normalize=True)
                
                # Split into pairs and score
                scores0 = (image_features[:batch_size] * text_features).sum(dim=-1).cpu().numpy()
                scores1 = (image_features[batch_size:] * text_features).sum(dim=-1).cpu().numpy()
                for i in range(batch_size):
                    results[i]["hps"] = np.array([scores0[i], scores1[i]], dtype=np.float32)
        except Exception as e:
            for i in range(batch_size):
                results[i]["hps"] = np.array([np.nan, np.nan], dtype=np.float32)
        
        return results


def process_single_shard(shard_name: str, scorer: BatchedUnifiedScorer, gpu_id: int):
    """Process a single WebDataset shard with batched inference and batch-level prefetching."""
    
    local_tar_path = hf_hub_download(repo_id=REPO_ID, filename=shard_name, repo_type=REPO_TYPE)
    dst_path = os.path.join(OUT_DIR, shard_name)
    tmp_path = dst_path + ".tmp"
    
    if FORCE_REWRITE:
        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)
        except Exception:
            pass
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
    else:
        # Skip shard if a non-empty .tar exists (legacy behavior)
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            return
        if os.path.exists(tmp_path):
            # Stale partial write from previous run; remove and regenerate
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    
    # Enable CUDA streams for async transfers
    torch.cuda.set_device(scorer.device)
    
    ds = wds.WebDataset(local_tar_path, handler=wds.ignore_and_continue, 
                       empty_check=False, shardshuffle=False).decode("pil")
    
    # Use batch-level prefetching: prefetch 4 complete batches in background
    # This hides ALL data loading latency (image decode, prompt parse, RGB conversion)
    # Supply a preprocessing function for CLIP/HPS (shared open_clip transforms live in CLIP/HPS selectors)
    preprocess_fn = None
    try:
        # Prefer CLIP's preprocess (HPS uses the same open_clip transforms)
        preprocess_fn = scorer.clip.preprocess_val
    except Exception:
        try:
            preprocess_fn = scorer.hps.preprocess_val
        except Exception:
            preprocess_fn = None

    batch_prefetcher = BatchPrefetcher(
        ds, batch_size=scorer.batch_size, prefetch_batches=8, preprocess_image_fn=preprocess_fn, pin_memory=True
    )
    
    successes = 0
    failures = 0
    total_processed = 0
    
    try:
        # Use AsyncTarWriter for non-blocking disk I/O
        with AsyncTarWriter(tmp_path, queue_size=20) as async_writer:
            pbar = tqdm(batch_prefetcher, desc=f"  GPU{gpu_id}:{shard_name}", unit="batches",
                       leave=False, position=gpu_id+1)
            
            for batch in pbar:
                batch_keys = batch['keys']
                batch_images_0 = batch['images_0']
                batch_images_1 = batch['images_1']
                batch_prompts = batch['prompts']
                batch_clip_t0 = batch.get('clip_t0')
                batch_clip_t1 = batch.get('clip_t1')
                
                try:
                    # Process batch on GPU (fast, keeps GPU at 100%)
                    # If preprocessed tensors exist, move them to device non-blockingly and pass through models
                    batch_results = None
                    if batch_clip_t0 is not None and batch_clip_t1 is not None and all(t is not None for t in batch_clip_t0 + batch_clip_t1):
                        # Prepare tensors for CLIP/HPS
                        with torch.no_grad():
                            clip_images = torch.stack(batch_clip_t0 + batch_clip_t1)
                            if torch.cuda.is_available():
                                clip_images = clip_images.to(device=scorer.device, non_blocking=True)

                        # Aesthetic & PickScore still consume PIL; CLIP/HPS will be recomputed using tensors below
                        # We'll call scorer.score_batch which already handles batched flows, but we will override
                        # CLIP/HPS preprocess path by temporarily setting helper attributes to avoid redundant CPU work.
                        pass
                    # Call scorer with optional preprocessed tensors
                    batch_results = scorer.score_batch(
                        batch_images_0,
                        batch_images_1,
                        batch_prompts,
                        clip_tensors0=batch_clip_t0,
                        clip_tensors1=batch_clip_t1,
                    )
                    
                    # Queue results for async writing (non-blocking!)
                    # Disk I/O happens in background while GPU processes next batch
                    for i, scores in enumerate(batch_results):
                        try:
                            out = {"__key__": batch_keys[i]}
                            out["pickscore.npy"] = array_to_npy_bytes(scores['pickscore'])
                            out["aesthetic.npy"] = array_to_npy_bytes(scores['aesthetic'])
                            out["clip_score.npy"] = array_to_npy_bytes(scores['clip'])
                            out["hps_score.npy"] = array_to_npy_bytes(scores['hps'])
                            async_writer.write(out)  # Non-blocking!
                            successes += 1
                        except Exception:
                            failures += 1
                except Exception:
                    failures += len(batch_keys)
                
                total_processed += len(batch_keys)
                pbar.set_postfix({"✓": successes, "✗": failures, "samples": total_processed})
            
            pbar.close()
            # Writer thread will flush all pending writes when context manager exits
        # Atomically move tmp shard into place only after successful close
        try:
            os.replace(tmp_path, dst_path)
        except Exception:
            # If replace fails, leave tmp for inspection; next run will clean it
            pass
    finally:
        batch_prefetcher.stop()
    
    tqdm.write(f"  GPU{gpu_id} ✓ {shard_name}: {successes} samples, {failures} failures")


def process_shard_list(shards: List[str], gpu_id: int):
    """Process a list of shards on a specific GPU."""
    device = f"cuda:{gpu_id}"
    
    # Enable cudnn autotuner for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    print(f"GPU {gpu_id}: Initializing models...")
    scorer = BatchedUnifiedScorer(device, batch_size=BATCH_SIZE)
    print(f"GPU {gpu_id}: Processing {len(shards)} shards")
    
    for shard in shards:
        process_single_shard(shard, scorer, gpu_id)
        # Clear cache between shards to prevent memory fragmentation
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("MULTI-GPU BATCHED ANNOTATION PIPELINE")
    print("=" * 70)
    
    # Get shard list
    all_shards = list_hf_shards(REPO_ID, REPO_TYPE)
    
    # Determine shard range
    if TARGET_SAMPLES is None:
        selected_shards = all_shards
    else:
        num_shards = math.ceil(TARGET_SAMPLES / SAMPLES_PER_SHARD)
        selected_shards = all_shards[:num_shards]
    
    # Apply manual shard range if specified
    if START_SHARD > 0 or END_SHARD is not None:
        end = END_SHARD if END_SHARD else len(selected_shards)
        selected_shards = selected_shards[START_SHARD:end]
        print(f"Manual shard range: {START_SHARD} to {end}")
    
    # Check for manual GPU mode
    if GPU_ID is not None:
        # Manual mode: single process for single GPU
        gpu_id = int(GPU_ID)
        print(f"Manual mode: Using GPU {gpu_id}")
        print(f"Processing shards: {START_SHARD} to {START_SHARD + len(selected_shards)}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Output: {OUT_DIR}")
        print("=" * 70)
        
        process_shard_list(selected_shards, gpu_id)
        
    else:
        # Auto mode: spawn processes for all GPUs
        gpus = get_available_gpus()
        
        if not gpus:
            print("No GPUs available! Using CPU (very slow)")
            device = "cpu"
            scorer = BatchedUnifiedScorer(device, batch_size=BATCH_SIZE)
            for shard in tqdm(selected_shards, desc="Processing"):
                process_single_shard(shard, scorer, 0)
        else:
            print(f"Available GPUs: {gpus}")
            print(f"Total shards: {len(selected_shards)}")
            print(f"Batch size per GPU: {BATCH_SIZE}")
            print(f"Output: {OUT_DIR}")
            print("=" * 70)
            
            # Split shards across GPUs
            shard_splits = split_shards(selected_shards, len(gpus))
            
            print("\nShard distribution:")
            for i, gpu_id in enumerate(gpus):
                print(f"  GPU {gpu_id}: {len(shard_splits[i])} shards")
            print()
            
            # Launch parallel processes
            import multiprocessing as mp
            mp.set_start_method('spawn', force=True)
            
            processes = []
            for i, gpu_id in enumerate(gpus):
                if not shard_splits[i]:
                    continue
                p = mp.Process(target=process_shard_list, 
                             args=(shard_splits[i], gpu_id))
                p.start()
                processes.append(p)
            
            # Wait for all processes
            for p in processes:
                p.join()
    
    print("=" * 70)
    print("✓ All GPUs completed!")
    print(f"Annotations saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()


# OPTIMIZED USAGE EXAMPLES:
# 
# High GPU utilization with optimized batch size:
# CUDA_VISIBLE_DEVICES=0,1,2 TARGET_SAMPLES=ALL BATCH_SIZE=32 python annotate_with_all_scores_multigpu.py
#
# For high-end GPUs (A100, H100) - use larger batches:
# CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=48 python annotate_with_all_scores_multigpu.py
#
# Monitor GPU utilization with: watch -n 0.5 nvidia-smi