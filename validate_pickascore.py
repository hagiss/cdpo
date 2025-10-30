import os
import sys
import json
import argparse
from typing import List, Optional, Tuple

import torch
from PIL import Image

from datasets import load_dataset

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
)

from cond_sd15 import (
    SD15ConditionAdapter,
    monkey_patch_sd15_pipeline_for_condition,
    monkey_patch_sd15_pipeline_for_ipadapter,
    IPAdapter,
    init_adapter,
)

from cond_sdxl import (
    SDXLConditionAdapter,
    IPAdapter_SDXL,
    init_adapter_SDXL,
    monkey_patch_sdxl_pipeline_for_ipadapter,
)

def detect_sdxl_from_model_name(model_name: str) -> bool:
    return 'stable-diffusion-xl' in model_name or 'sdxl' in model_name.lower()


def build_pipelines(
    pretrained_model_name: str,
    dpo_ckpt_path: Optional[str],
    device: str = 'cuda',
    torch_dtype: torch.dtype = torch.float16,
    disable_safety_checker: bool = True,
    cond_adapter_subpath: Optional[str] = None,
    cond_projector_type: str = "linear",
    cond_mlp_hidden_dim: int = 4096,
    enable_ipadapter: bool = False,
    ipadapter_ckpt: Optional[str] = None,
    cond_positive_text: str = "win",
    cond_negative_text: str = "lose",
) -> Tuple[object, Optional[object]]:
    is_sdxl = detect_sdxl_from_model_name(pretrained_model_name)

    if is_sdxl:
        pipe_base = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name, torch_dtype=torch_dtype, variant="fp16", use_safetensors=True
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
        )
        pipe_base.vae = vae
    else:
        pipe_base = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name, torch_dtype=torch_dtype
        )
        # unet_id = "models/dpo-sd1.5"
        # unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
        # pipe_base.unet = unet

    pipe_base = pipe_base.to(device)
    if disable_safety_checker:
        try:
            pipe_base.safety_checker = None
        except Exception:
            pass

    pipe_dpo = None
    # Build a second pipeline if we have a DPO checkpoint OR if IP-Adapter sampling is requested
    if (dpo_ckpt_path is not None) or enable_ipadapter:
        if is_sdxl:
            pipe_dpo = StableDiffusionXLPipeline.from_pretrained(
                pretrained_model_name, torch_dtype=torch_dtype, variant="fp16", use_safetensors=True
            )
        else:
            pipe_dpo = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name, torch_dtype=torch_dtype
            )
        pipe_dpo = pipe_dpo.to(device)
        if disable_safety_checker:
            try:
                pipe_dpo.safety_checker = None
            except Exception:
                pass

        # Replace UNet with trained weights when DPO checkpoint is provided
        if dpo_ckpt_path is not None:
            dpo_unet = UNet2DConditionModel.from_pretrained(
                dpo_ckpt_path, subfolder='unet', torch_dtype=torch_dtype
            ).to(device)
            pipe_dpo.unet = dpo_unet

        # SD1.5-only conditional & IP-Adapter logic
        if not is_sdxl:
            # IP-Adapter takes precedence if enabled (wrap UNet and patch special __call__)
            if enable_ipadapter and pipe_dpo is not None:
                # Build image projection model (prefer loading cond_adapter config if available)
                image_proj_model = None
                ca_path = None
                if cond_adapter_subpath and os.path.isdir(cond_adapter_subpath):
                    ca_path = cond_adapter_subpath
                elif dpo_ckpt_path and os.path.isdir(os.path.join(dpo_ckpt_path, 'cond_adapter')):
                    ca_path = os.path.join(dpo_ckpt_path, 'cond_adapter')
                if ca_path is not None and os.path.isdir(ca_path):
                    try:
                        image_proj_model = SD15ConditionAdapter.from_pretrained(ca_path, projector_type=cond_projector_type, mlp_hidden_dim=cond_mlp_hidden_dim).to(device)
                    except Exception as e:
                        print(f"[WARN] Failed to load cond_adapter at {ca_path}: {e}. Falling back to default SD15ConditionAdapter().")
                if image_proj_model is None:
                    image_proj_model = SD15ConditionAdapter(projector_type=cond_projector_type, mlp_hidden_dim=cond_mlp_hidden_dim).to(device)

                # Install attention processors and wrap UNet
                adapter_modules = init_adapter(pipe_dpo.unet)
                ip_ckpt_root = ipadapter_ckpt if ipadapter_ckpt else dpo_ckpt_path
                ip_unet = IPAdapter(pipe_dpo.unet, image_proj_model, adapter_modules, ckpt_path=ip_ckpt_root).to(device)
                pipe_dpo.unet = ip_unet
                # Patch pipeline to route cond texts during __call__
                monkey_patch_sd15_pipeline_for_ipadapter(pipe_dpo)
            else:
                # Fallback: conditional adapter for conditioned CFG
                adapter_path = cond_adapter_subpath if cond_adapter_subpath else os.path.join(dpo_ckpt_path, 'cond_adapter')
                if os.path.isdir(adapter_path):
                    try:
                        cond_adapter = SD15ConditionAdapter.from_pretrained(adapter_path).to(device)
                        monkey_patch_sd15_pipeline_for_condition(
                            pipe_dpo,
                            cond_adapter,
                            positive_condition=cond_positive_text,
                            negative_condition=cond_negative_text,
                        )
                    except Exception as e:
                        print(f"[WARN] Failed to load/patch conditional adapter at {adapter_path}: {e}")
        elif is_sdxl:
            if enable_ipadapter and pipe_dpo is not None:
                image_proj_model = None
                ca_path = None
                if cond_adapter_subpath and os.path.isdir(cond_adapter_subpath):
                    ca_path = cond_adapter_subpath
                elif dpo_ckpt_path and os.path.isdir(os.path.join(dpo_ckpt_path, 'cond_adapter')):
                    ca_path = os.path.join(dpo_ckpt_path, 'cond_adapter')
                if ca_path is not None and os.path.isdir(ca_path):
                    try:
                        image_proj_model = SDXLConditionAdapter.from_pretrained(ca_path, projector_type=cond_projector_type, mlp_hidden_dim=cond_mlp_hidden_dim).to(device)
                    except Exception as e:
                        print(f"[WARN] Failed to load cond_adapter at {ca_path}: {e}. Falling back to default SDXLConditionAdapter().")
                if image_proj_model is None:
                    image_proj_model = SDXLConditionAdapter(projector_type=cond_projector_type, mlp_hidden_dim=cond_mlp_hidden_dim).to(device)

                adapter_modules = init_adapter_SDXL(pipe_dpo.unet)
                ip_ckpt_root = ipadapter_ckpt if ipadapter_ckpt else dpo_ckpt_path
                ip_unet = IPAdapter_SDXL(pipe_dpo.unet, image_proj_model, adapter_modules, ckpt_path=ip_ckpt_root).to(device)
                pipe_dpo.unet = ip_unet
                monkey_patch_sdxl_pipeline_for_ipadapter(pipe_dpo)

    return pipe_base, pipe_dpo


def load_prompts_from_dataset(
    dataset_name: str = 'kashif/pickascore',
    split: str = 'validation',
    limit: Optional[int] = None,
) -> List[str]:
    ds = load_dataset(dataset_name, split=split)
    # Try common prompt field names
    candidate_fields = ['caption', 'prompt', 'text', 'Prompt']
    field = None
    for f in candidate_fields:
        if f in ds.column_names:
            field = f
            break
    if field is None:
        raise KeyError(f"No prompt field found in dataset columns: {ds.column_names}")
    prompts = list(ds[field])
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def load_prompts_from_file(path: str, limit: Optional[int] = None) -> List[str]:
    prompts: List[str] = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                # Support keys: caption, prompt, text
                for key in ['caption', 'prompt', 'text', 'Prompt']:
                    if key in obj:
                        prompts.append(str(obj[key]))
                        break
                if limit is not None and len(prompts) >= limit:
                    break
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
                if limit is not None and len(prompts) >= limit:
                    break
    return prompts


def save_prompts_to_file(prompts: List[str], out_path: str, jsonl: bool = False) -> None:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if jsonl:
        with open(out_path, 'w', encoding='utf-8') as f:
            for idx, p in enumerate(prompts):
                f.write(json.dumps({"id": idx, "caption": p}, ensure_ascii=False) + "\n")
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            for p in prompts:
                f.write(p + "\n")


def generate_image(pipe, prompt: str, seed: int, guidance_scale: float, guidance_rescale: float = 0.0, reference_conditional_guidance: bool = False, decomposed_additive_guidance: bool = False, extra_kwargs: Optional[dict] = None) -> Image.Image:
    generator = torch.Generator(device=pipe.device)
    generator = generator.manual_seed(seed)
    call_kwargs = dict(prompt=prompt, generator=generator, guidance_scale=guidance_scale, guidance_rescale=guidance_rescale)
    if extra_kwargs:
        call_kwargs.update(extra_kwargs)
    # Check whether unet's class is IPAdapter
    is_ipadapter = pipe.unet.__class__.__name__ == "IPAdapter" or pipe.unet.__class__.__name__ == "IPAdapter_SDXL"
    # You can use is_ipadapter as needed, e.g., for debugging or conditional logic
    if is_ipadapter:
        image = pipe.__call__(self=pipe, decomposed_additive_guidance=decomposed_additive_guidance, **call_kwargs).images[0]
    else:
        image = pipe(**call_kwargs).images[0]
    return image


def cmd_download_prompts(args: argparse.Namespace) -> None:
    prompts = load_prompts_from_dataset(
        dataset_name='kashif/pickascore',
        split=args.split,
        limit=args.limit,
    )
    save_prompts_to_file(prompts, args.out, jsonl=args.jsonl)
    print(f"Saved {len(prompts)} prompts to {args.out}")


def cmd_sample(args: argparse.Namespace) -> None:
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    pipe_base, pipe_dpo = build_pipelines(
        pretrained_model_name=args.pretrained_model_name,
        dpo_ckpt_path=args.ckpt_path if not args.baseline_only else None,
        device=device,
        torch_dtype=torch.float16,
        cond_adapter_subpath=args.cond_adapter,
        cond_projector_type=args.cond_projector_type,
        cond_mlp_hidden_dim=args.cond_mlp_hidden_dim,
        enable_ipadapter=(args.ipadapter_ckpt is not None),
        ipadapter_ckpt=args.ipadapter_ckpt,
        cond_positive_text=args.cond_positive_text,
        cond_negative_text=args.cond_negative_text,
    )

    # Load prompts
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file, limit=args.limit)
    else:
        prompts = load_prompts_from_dataset(limit=args.limit)

    # Prepare output folders
    # If ckpt_path is provided, do NOT sample baseline (only DPO)
    sample_baseline = (args.ckpt_path is None) and (args.ipadapter_ckpt is None)
    sample_dpo = (args.ckpt_path is not None or args.ipadapter_ckpt is not None) and pipe_dpo is not None

    base_dir = os.path.join(args.out_dir, 'baseline')
    dpo_dir = os.path.join(args.out_dir, 'dpo')
    os.makedirs(args.out_dir, exist_ok=True)
    if sample_baseline:
        os.makedirs(base_dir, exist_ok=True)
    if sample_dpo:
        os.makedirs(dpo_dir, exist_ok=True)

    # Save prompts index mapping
    with open(os.path.join(args.out_dir, 'prompts.jsonl'), 'w', encoding='utf-8') as f:
        for idx, p in enumerate(prompts):
            f.write(json.dumps({"id": idx, "caption": p}, ensure_ascii=False) + "\n")

    guidance_scale = args.guidance_scale
    guidance_rescale = args.guidance_rescale
    seed = args.seed

    for idx, prompt in enumerate(prompts):
        # Keep seed identical for both for fair comparison
        sample_seed = seed if seed is not None else 0

        # Baseline (skip if ckpt_path provided)
        if sample_baseline:
            img_base = generate_image(pipe_base, prompt, seed=sample_seed, guidance_scale=guidance_scale, guidance_rescale=guidance_rescale)
            img_base.save(os.path.join(base_dir, f"{idx:06d}.png"))

        # DPO (if ckpt_path or ipadapter_ckpt provided)
        if sample_dpo:
            extra_kwargs = None
            if args.ipadapter_ckpt is not None:
                # Pass condition texts for IP-Adapter call
                extra_kwargs = {
                    "positive_condition": args.cond_positive_text,
                    "negative_condition": args.cond_negative_text,
                }
            img_dpo = generate_image(
                pipe_dpo,
                prompt,
                seed=sample_seed,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                reference_conditional_guidance=args.reference_conditional_guidance,
                decomposed_additive_guidance=args.decomposed_additive_guidance,
                extra_kwargs=extra_kwargs,
            )
            img_dpo.save(os.path.join(dpo_dir, f"{idx:06d}.png"))

        if (idx + 1) % max(1, args.log_every) == 0:
            print(f"Generated {idx + 1}/{len(prompts)}")

    print("Sampling complete.")


def load_prompt_list_for_eval(prompts_path: str) -> List[str]:
    return load_prompts_from_file(prompts_path)


def get_sorted_image_paths(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    names = [n for n in os.listdir(folder) if n.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    names.sort()
    return [os.path.join(folder, n) for n in names]


def safe_import_selectors(device: str):
    # PickScore is required
    from utils.pickscore_utils import Selector as PickSelector
    ps_selector = PickSelector(device)

    # AES optional
    aes_selector = None
    try:
        from utils.aes_utils import Selector as AESSelector
        aes_selector = AESSelector(device)
    except Exception as e:
        print(f"[WARN] AES selector unavailable: {e}")

    # HPS optional
    hps_selector = None
    try:
        from utils.hps_utils import Selector as HPSSelector
        hps_selector = HPSSelector(device)
    except Exception as e:
        print(f"[WARN] HPS selector unavailable: {e}")

    # CLIP optional
    clip_selector = None
    try:
        from utils.clip_utils import Selector as CLIPSelector
        clip_selector = CLIPSelector(device)
    except Exception as e:
        print(f"[WARN] CLIP selector unavailable: {e}")

    # ImageReward optional
    imagereward_selector = None
    try:
        import ImageReward as RM
        class ImageRewardSelector:
            def __init__(self):
                self.model = RM.load("ImageReward-v1.0")
            
            def score(self, images, prompt):
                # Convert PIL images to the format ImageReward expects
                return self.model.score(prompt, images)
        
        imagereward_selector = ImageRewardSelector()
    except Exception as e:
        print(f"[WARN] ImageReward selector unavailable: {e}")

    return ps_selector, aes_selector, hps_selector, clip_selector, imagereward_selector


def fmt_mean(sums, counts, idx: int):
    if sums is None or counts is None:
        return None
    return (sums[idx] / counts[idx]) if counts[idx] > 0 else None


def fmt_winrate(wins: Optional[int], losses: Optional[int]):
    total = (wins or 0) + (losses or 0)
    return (wins / total) if total > 0 else None


def cmd_eval_folders(args: argparse.Namespace) -> None:
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    ps_selector, aes_selector, hps_selector, clip_selector, imagereward_selector = safe_import_selectors(device)

    # Set default output path if not specified (same directory as baseline_folder parent)
    if args.output is None:
        parent_dir = os.path.dirname(os.path.abspath(args.dpo_folder))
        args.output = os.path.join(parent_dir, 'results.txt')

    prompts = load_prompt_list_for_eval(args.prompts)
    base_paths = get_sorted_image_paths(args.baseline_folder)
    dpo_paths = get_sorted_image_paths(args.dpo_folder)

    n = min(len(prompts), len(base_paths), len(dpo_paths))
    prompts = prompts[:n]
    base_paths = base_paths[:n]
    dpo_paths = dpo_paths[:n]

    ps_sums = [0.0, 0.0]
    ps_counts = [0, 0]
    aes_sums = [0.0, 0.0] if aes_selector is not None else None
    aes_counts = [0, 0] if aes_selector is not None else None
    hps_sums = [0.0, 0.0] if hps_selector is not None else None
    hps_counts = [0, 0] if hps_selector is not None else None
    clip_sums = [0.0, 0.0] if clip_selector is not None else None
    clip_counts = [0, 0] if clip_selector is not None else None
    clip_loss_indices = [] if clip_selector is not None else None
    imagereward_sums = [0.0, 0.0] if imagereward_selector is not None else None
    imagereward_counts = [0, 0] if imagereward_selector is not None else None

    ps_wins = 0
    ps_losses = 0
    ps_ties = 0
    aes_wins = 0 if aes_selector is not None else None
    aes_losses = 0 if aes_selector is not None else None
    aes_ties = 0 if aes_selector is not None else None
    hps_wins = 0 if hps_selector is not None else None
    hps_losses = 0 if hps_selector is not None else None
    hps_ties = 0 if hps_selector is not None else None
    clip_wins = 0 if clip_selector is not None else None
    clip_losses = 0 if clip_selector is not None else None
    clip_ties = 0 if clip_selector is not None else None
    imagereward_wins = 0 if imagereward_selector is not None else None
    imagereward_losses = 0 if imagereward_selector is not None else None
    imagereward_ties = 0 if imagereward_selector is not None else None

    for idx in range(n):
        prompt = prompts[idx]
        # Load images with PIL for PickScore/AES; HPS can take PIL too
        im_base = Image.open(base_paths[idx]).convert('RGB')
        im_dpo = Image.open(dpo_paths[idx]).convert('RGB')
        ims = [im_base, im_dpo]

        # PickScore
        ps_scores = ps_selector.score(ims, prompt)
        for i, s in enumerate(ps_scores[:2]):
            ps_sums[i] += float(s)
            ps_counts[i] += 1
        if len(ps_scores) >= 2:
            base_ps = float(ps_scores[0])
            dpo_ps = float(ps_scores[1])
            if dpo_ps > base_ps + 1e-8:
                ps_wins += 1
            elif base_ps > dpo_ps + 1e-8:
                ps_losses += 1
            else:
                ps_ties += 1

        # AES
        if aes_selector is not None:
            try:
                aes_scores = aes_selector.score(ims, "")
                for i, s in enumerate(aes_scores[:2]):
                    aes_sums[i] += float(s)
                    aes_counts[i] += 1
                if len(aes_scores) >= 2:
                    base_aes = float(aes_scores[0])
                    dpo_aes = float(aes_scores[1])
                    if dpo_aes > base_aes + 1e-8:
                        aes_wins += 1
                    elif base_aes > dpo_aes + 1e-8:
                        aes_losses += 1
                    else:
                        aes_ties += 1
            except Exception as e:
                print(f"[WARN] AES scoring failed at index {idx}: {e}")

        # HPS
        if hps_selector is not None:
            try:
                hps_scores = hps_selector.score(ims, prompt)
                for i, s in enumerate(hps_scores[:2]):
                    hps_sums[i] += float(s)
                    hps_counts[i] += 1
                if len(hps_scores) >= 2:
                    base_hps = float(hps_scores[0])
                    dpo_hps = float(hps_scores[1])
                    if dpo_hps > base_hps + 1e-8:
                        hps_wins += 1
                    elif base_hps > dpo_hps + 1e-8:
                        hps_losses += 1
                    else:
                        hps_ties += 1
            except Exception as e:
                print(f"[WARN] HPS scoring failed at index {idx}: {e}")

        # CLIP
        if clip_selector is not None:
            try:
                clip_scores = clip_selector.score(ims, prompt)
                for i, s in enumerate(clip_scores[:2]):
                    clip_sums[i] += float(s)
                    clip_counts[i] += 1
                if len(clip_scores) >= 2:
                    base_clip = float(clip_scores[0])
                    dpo_clip = float(clip_scores[1])
                    if dpo_clip > base_clip + 1e-8:
                        clip_wins += 1
                    elif base_clip > dpo_clip + 1e-8:
                        clip_losses += 1
                        if clip_loss_indices is not None:
                            clip_loss_indices.append(idx)
                    else:
                        clip_ties += 1
            except Exception as e:
                print(f"[WARN] CLIP scoring failed at index {idx}: {e}")

        # ImageReward
        if imagereward_selector is not None:
            try:
                imagereward_scores = imagereward_selector.score(ims, prompt)
                for i, s in enumerate(imagereward_scores[:2]):
                    imagereward_sums[i] += float(s)
                    imagereward_counts[i] += 1
                if len(imagereward_scores) >= 2:
                    base_imagereward = float(imagereward_scores[0])
                    dpo_imagereward = float(imagereward_scores[1])
                    if dpo_imagereward > base_imagereward + 1e-8:
                        imagereward_wins += 1
                    elif base_imagereward > dpo_imagereward + 1e-8:
                        imagereward_losses += 1
                    else:
                        imagereward_ties += 1
            except Exception as e:
                print(f"[WARN] ImageReward scoring failed at index {idx}: {e}")

        if (idx + 1) % max(1, args.log_every) == 0:
            print(f"Scored {idx + 1}/{n}")

    # Prepare results text
    results_lines = []
    results_lines.append("==== Mean scores across prompts ====")
    results_lines.append(f"PickScore mean - Baseline: {fmt_mean(ps_sums, ps_counts, 0)}, DPO: {fmt_mean(ps_sums, ps_counts, 1)}")
    if aes_sums is not None:
        results_lines.append(f"AES mean - Baseline: {fmt_mean(aes_sums, aes_counts, 0)}, DPO: {fmt_mean(aes_sums, aes_counts, 1)}")
    else:
        results_lines.append("AES mean - skipped (AES selector unavailable)")
    if hps_sums is not None:
        results_lines.append(f"HPS mean - Baseline: {fmt_mean(hps_sums, hps_counts, 0)}, DPO: {fmt_mean(hps_sums, hps_counts, 1)}")
    else:
        results_lines.append("HPS mean - skipped (HPS selector unavailable)")
    if clip_sums is not None:
        results_lines.append(f"CLIP mean - Baseline: {fmt_mean(clip_sums, clip_counts, 0)}, DPO: {fmt_mean(clip_sums, clip_counts, 1)}")
    else:
        results_lines.append("CLIP mean - skipped (CLIP selector unavailable)")
    if imagereward_sums is not None:
        results_lines.append(f"ImageReward mean - Baseline: {fmt_mean(imagereward_sums, imagereward_counts, 0)}, DPO: {fmt_mean(imagereward_sums, imagereward_counts, 1)}")
    else:
        results_lines.append("ImageReward mean - skipped (ImageReward selector unavailable)")

    results_lines.append("")
    results_lines.append("==== Win rates (DPO vs Baseline) ====")
    results_lines.append(f"PickScore win-rate: {fmt_winrate(ps_wins, ps_losses)} (wins={ps_wins}, losses={ps_losses}, ties={ps_ties})")
    if aes_sums is not None:
        results_lines.append(f"AES win-rate: {fmt_winrate(aes_wins, aes_losses)} (wins={aes_wins}, losses={aes_losses}, ties={aes_ties})")
    else:
        results_lines.append("AES win-rate - skipped (AES selector unavailable)")
    if hps_sums is not None:
        results_lines.append(f"HPS win-rate: {fmt_winrate(hps_wins, hps_losses)} (wins={hps_wins}, losses={hps_losses}, ties={hps_ties})")
    else:
        results_lines.append("HPS win-rate - skipped (HPS selector unavailable)")
    if clip_sums is not None:
        results_lines.append(f"CLIP win-rate: {fmt_winrate(clip_wins, clip_losses)} (wins={clip_wins}, losses={clip_losses}, ties={clip_ties})")
    else:
        results_lines.append("CLIP win-rate - skipped (CLIP selector unavailable)")
    if imagereward_sums is not None:
        results_lines.append(f"ImageReward win-rate: {fmt_winrate(imagereward_wins, imagereward_losses)} (wins={imagereward_wins}, losses={imagereward_losses}, ties={imagereward_ties})")
    else:
        results_lines.append("ImageReward win-rate - skipped (ImageReward selector unavailable)")

    # Print to console
    print()
    for line in results_lines:
        print(line)

    # Save to file
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_lines) + '\n')
    print(f"\nResults saved to: {args.output}")

    if clip_loss_indices is not None:
        output_dir = os.path.dirname(args.output) or '.'
        indices_path = os.path.join(output_dir, 'clip_dpo_losses_indices.json')
        with open(indices_path, 'w', encoding='utf-8') as f:
            json.dump(clip_loss_indices, f)
        print(f"Indices where baseline CLIP score was higher saved to: {indices_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Validate models on kashif/pickascore')
    sub = p.add_subparsers(dest='cmd', required=True)

    # download-prompts
    p_dl = sub.add_parser('download-prompts', help='Download prompts from kashif/pickascore')
    p_dl.add_argument('--split', type=str, default='validation')
    p_dl.add_argument('--limit', type=int, default=None)
    p_dl.add_argument('--out', type=str, required=True, help='Output file (.txt or .jsonl)')
    p_dl.add_argument('--jsonl', action='store_true', help='Save as JSONL with {id, caption}')
    p_dl.set_defaults(func=cmd_download_prompts)

    # sample
    p_s = sub.add_parser('sample', help='Sample images for baseline and optionally DPO model')
    p_s.add_argument('--pretrained-model-name', type=str, default='runwayml/stable-diffusion-v1-5')
    p_s.add_argument('--ckpt-path', type=str, default=None, help='Path to trained checkpoint folder containing unet/')
    p_s.add_argument('--cond-adapter', type=str, default=None, help='Optional path to cond_adapter (defaults to <ckpt>/cond_adapter)')
    p_s.add_argument("--cond-projector-type", type=str, default="linear", choices=["linear", "mlp"], help="Condition adapter projector type")
    p_s.add_argument("--cond-mlp-hidden-dim", type=int, default=4096, help="Hidden dim for MLP projector (if used)")
    p_s.add_argument('--ipadapter-ckpt', type=str, default=None, help='Enable IP-Adapter and load from this checkpoint (expects ip_adapter/ subfolder)')
    p_s.add_argument('--cond-positive-text', type=str, default='win')
    p_s.add_argument('--cond-negative-text', type=str, default='lose')
    p_s.add_argument('--prompts-file', type=str, default=None, help='Path to prompts (.txt or .jsonl). If omitted, loads from dataset')
    p_s.add_argument('--limit', type=int, default=None, help='Max prompts to sample')
    p_s.add_argument('--out-dir', type=str, required=True)
    p_s.add_argument('--seed', type=int, default=0)
    p_s.add_argument('--guidance-scale', type=float, default=7.5)
    p_s.add_argument('--guidance-rescale', type=float, default=0.0)
    p_s.add_argument('--reference-conditional-guidance', action='store_true', help='Use reference conditional guidance')
    p_s.add_argument('--decomposed-additive-guidance', action='store_true', help='Use decomposed additive guidance')
    p_s.add_argument('--baseline-only', action='store_true', help='Only sample baseline images')
    p_s.add_argument('--log-every', type=int, default=25)
    p_s.add_argument('--cpu', action='store_true')
    p_s.set_defaults(func=cmd_sample)

    # eval-folders
    p_e = sub.add_parser('eval-folders', help='Evaluate two folders (baseline vs dpo) with PickScore/AES/HPS')
    p_e.add_argument('--prompts', type=str, required=True, help='Prompts file used for generation (.txt or .jsonl)')
    p_e.add_argument('--baseline-folder', type=str, required=True)
    p_e.add_argument('--dpo-folder', type=str, required=True)
    p_e.add_argument('--output', type=str, default=None, help='Path to save evaluation results as text file (default: results.txt in parent dir of baseline-folder)')
    p_e.add_argument('--log-every', type=int, default=50)
    p_e.add_argument('--cpu', action='store_true')
    p_e.set_defaults(func=cmd_eval_folders)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    torch.set_grad_enabled(False)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()


