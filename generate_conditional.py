#!/usr/bin/env python3

import argparse
import os
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel

from cond_sd15 import SD15ConditionAdapter, monkey_patch_sd15_pipeline_for_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with SD1.5 (or SDXL) using a trained UNet and conditional adapter.")
    parser.add_argument("--pretrained_model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model repo or local path.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained checkpoint folder (expects subfolder 'unet').")
    parser.add_argument("--cond_adapter_path", type=str, default=None, help="Path to conditional adapter folder (contains config.json and pytorch_model.bin). If omitted, tries '<checkpoint>/cond_adapter'.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative text prompt.")
    parser.add_argument("--positive_condition", type=str, default="win", help="Positive condition text.")
    parser.add_argument("--negative_condition", type=str, default="lose", help="Negative condition text.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale (defaults to 7.5 for SD1.5, 3.5 for SDXL).")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 where possible.")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path.")
    return parser.parse_args()


def build_pipeline(pretrained_model_name: str, device: str, dtype: torch.dtype):
    is_sdxl = ("stable-diffusion-xl" in pretrained_model_name or "sdxl" in pretrained_model_name.lower())
    if is_sdxl:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=dtype,
        )
    pipe = pipe.to(device)
    # Disable safety checker for reproducibility
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    return pipe, is_sdxl


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)

    dtype = torch.float16 if args.fp16 and args.device == "cuda" else torch.float32

    pipe, is_sdxl = build_pipeline(args.pretrained_model_name, args.device, dtype)

    # Determine default guidance scale
    if args.guidance_scale is None:
        guidance_scale = 3.5 if is_sdxl else 7.5
    else:
        guidance_scale = args.guidance_scale

    # Load trained UNet (from checkpoint_path/unet)
    unet = UNet2DConditionModel.from_pretrained(
        args.checkpoint_path,
        subfolder="unet",
        torch_dtype=dtype,
    ).to(args.device)
    pipe.unet = unet

    # Load conditional adapter and patch the pipeline
    cond_adapter_dir: Optional[str] = args.cond_adapter_path
    if cond_adapter_dir is None:
        candidate = os.path.join(args.checkpoint_path, "cond_adapter")
        cond_adapter_dir = candidate if os.path.isdir(candidate) else None

    if cond_adapter_dir is not None:
        try:
            cond_adapter = SD15ConditionAdapter.from_pretrained(cond_adapter_dir).to(args.device)
            monkey_patch_sd15_pipeline_for_condition(
                pipe,
                cond_adapter,
                positive_condition=args.positive_condition,
                negative_condition=args.negative_condition,
            )
            print(f"[INFO] Conditional adapter loaded from '{cond_adapter_dir}' and pipeline patched.")
        except Exception as e:
            print(f"[WARN] Failed to load/patch conditional adapter from '{cond_adapter_dir}': {e}")
    else:
        print("[INFO] No conditional adapter path provided and none found; proceeding without conditioning patch.")

    # Build generator for deterministic sampling
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # Execute generation
    if is_sdxl:
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=guidance_scale,
            height=args.height,
            width=args.width,
        )
    else:
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=guidance_scale,
            height=args.height,
            width=args.width,
        )

    image = result.images[0]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image.save(args.output)
    print(f"[OK] Saved image to {args.output}")


if __name__ == "__main__":
    main()
