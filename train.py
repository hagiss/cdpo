#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import io
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,     StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

# Conditional SD1.5 adapter (monkey patch 1)
from cond_sd15 import SD15ConditionAdapter, monkey_patch_sd15_pipeline_for_condition, monkey_patch_sd15_pipeline_for_ipadapter, IPAdapter, init_adapter

if is_wandb_available():
    import wandb
    wandb.login(key="881a9b14affd90a6fe2d60376ba0f08be5a6bee8")

    
    
## SDXL
import functools
import gc
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, PretrainedConfig



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"), # label_0 is 1 means jpg_0 wins
    "hagiss/mvv_full": ("jpg_0", "jpg_1", "label_0", "caption", "prompt", 'mps_probs', 'vqa_scores', 'vila_scores'), # label_0 is 0 means jpg_0 wins
}

# Fixed sample prompts for qualitative monitoring during training
SAMPLE_PROMPTS = [
    "A pile of sand swirling in the wind forming the shape of a dancer",
    "A giant dinosaur frozen into a glacier and recently discovered by scientists, cinematic still",
    "a smiling beautiful sorceress with long dark hair and closed eyes wearing a dark top surrounded by glowing fire sparks at night, magical light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
    "A purple raven flying over big sur, light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
]

# MAX_MPS = 1.0
MAX_MPS = 27.53125
MAX_VQA = 0.9924590587615967
MAX_VILIA = 0.8928629159927368

MAX_PICK = 0.2825494408607483
MAX_AES = 8.049736022949219
MAX_CLIP = 0.60016268491745
MAX_HPS = 0.34084761142730713

# MIN_MPS = 0.00015723705291748047
MIN_MPS = -11.46875
MIN_VQA = 0.03758121654391289
MIN_VILIA = 0.23353318870067596

MIN_PICK = 0.1148335188627243
MIN_AES = 2.0321502685546875
MIN_CLIP = -0.14597028493881226
MIN_HPS = 0.15893374383449554

def normalize_mps(mps):
    norm = (mps - MIN_MPS) / (MAX_MPS - MIN_MPS)
    return int(round(norm * 4 + 1))

def normalize_vqa(vqa):
    norm = (vqa - MIN_VQA) / (MAX_VQA - MIN_VQA)
    return int(round(norm * 4 + 1))

def normalize_vila(vila):
    norm = (vila - MIN_VILIA) / (MAX_VILIA - MIN_VILIA)
    return int(round(norm * 4 + 1))

def normalize_pick(pick):
    norm = (pick - MIN_PICK) / (MAX_PICK - MIN_PICK)
    return int(round(norm * 20 + 1))

def normalize_aes(aes):
    norm = (aes - MIN_AES) / (MAX_AES - MIN_AES)
    return int(round(norm * 20 + 1))

def normalize_clip(clip):
    norm = (clip - MIN_CLIP) / (MAX_CLIP - MIN_CLIP)
    return int(round(norm * 20 + 1))

def normalize_hps(hps):
    norm = (hps - MIN_HPS) / (MAX_HPS - MIN_HPS)
    return int(round(norm * 20 + 1))

# def normalize_mps(mps):
#     return mps

# def normalize_vqa(vqa):
#     return vqa

# def normalize_vila(vila):
#     return vila
        
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Whether or not to use adafactor (should save mem)"
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    ## SDXL
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--sdxl", action='store_true', help="Train sdxl")
    
    ## DPO
    parser.add_argument("--sft", action='store_true', help="Run Supervised Fine-Tuning instead of Direct Preference Optimization")
    parser.add_argument("--beta_dpo", type=float, default=5000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    )
    parser.add_argument(
        "--unet_init", type=str, default='', help="Initialize start of run from unet (not compatible w/ checkpoint load)"
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    parser.add_argument(
        "--choice_model", type=str, default='', help="Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore"
    )
    parser.add_argument(
        "--dreamlike_pairs_only", action="store_true", help="Only train on pairs where both generations are from dreamlike"
    )
    parser.add_argument(
        "--scores_mapping_file", type=str, default=None, help="Path to JSON/pickle file mapping keys to scores (pickscore, aesthetic, clip_score, hps_score). Scores are loaded on-the-fly during training."
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming mode to avoid downloading entire dataset (saves disk space)"
    )
    # Conditional training/inference (SD1.5)
    parser.add_argument("--train_method", type=str, default=None, choices=["sft", "dpo", "csft", "cdpo"], help="Training method: sft/dpo/csft/cdpo")
    parser.add_argument("--csft", action='store_true', help="Alias for --train_method csft")
    parser.add_argument("--cdpo", action='store_true', help="Alias for --train_method cdpo")
    parser.add_argument("--cond_projector_type", type=str, default="linear", choices=["linear", "mlp"], help="Condition adapter projector type")
    parser.add_argument("--cond_mlp_hidden_dim", type=int, default=4096, help="Hidden dim for MLP projector (if used)")
    parser.add_argument("--cond_num_tokens", type=int, default=1, help="Deprecated")
    parser.add_argument("--cond_positive_text", type=str, default="win", help="Positive condition text")
    parser.add_argument("--cond_negative_text", type=str, default="lose", help="Negative condition text")
    parser.add_argument("--csft_cond_only", action='store_true', help="In CSFT, freeze UNet and train only conditional adapter")
    parser.add_argument("--ip_adapter", action='store_true', help="Use IP adapter")
    parser.add_argument("--ip_adapter_ckpt", type=str, default=None, help="Path to IP adapter checkpoint")
    parser.add_argument("--simultaneous_conditioning", action='store_true', help="Use simultaneous conditioning")
    parser.add_argument("--jeremy_conditioning", action='store_true', help="Use Jeremy conditioning")
    parser.add_argument("--class_conditioning", action='store_true', help="Add learned class embeddings for win/lose conditions")
    parser.add_argument("--multi_dim", action='store_true', help="Use multi-dimensional conditioning")
    # Initialize conditional adapter from a previous run (e.g., trained with --csft_cond_only)
    parser.add_argument(
        "--cond_adapter_init",
        type=str,
        default=None,
        help=(
            "Path to a pretrained conditional adapter directory to initialize from. "
            "Can point directly to the adapter folder containing config.json & pytorch_model.bin, "
            "or to a parent directory that contains a 'cond_adapter' subfolder."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    ## SDXL
    if args.sdxl:
        print("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512
            
    # Resolve training method
    if args.train_method is not None:
        pass
    elif args.csft:
        args.train_method = 'csft'
    elif args.cdpo:
        args.train_method = 'cdpo'
    else:
        args.train_method = 'sft' if args.sft else 'dpo'
    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to('cuda'),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}




def main():
    
    args = parse_args()
    
    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[
            accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=True)
        ],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    
    
    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, 
                                                    subfolder="scheduler")
    def enforce_zero_terminal_snr(scheduler):
        # Modified from https://arxiv.org/pdf/2305.08891.pdf
        # Turbo needs zero terminal SNR to truly learn from noise
        # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
    
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return 
    if 'turbo' in args.pretrained_model_name_or_path:
        enforce_zero_terminal_snr(noise_scheduler)
    
    # SDXL has two text encoders
    if args.sdxl:
        # Load the tokenizers
        if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
            tokenizer_and_encoder_name = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            tokenizer_and_encoder_name = args.pretrained_model_name_or_path
        tokenizer_one = AutoTokenizer.from_pretrained(
            tokenizer_and_encoder_name, subfolder="tokenizer", revision=args.revision, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

    # Not sure if we're hitting this at all
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    
    # BRAM NOTE: We're not using deepspeed currently so not sure it'll work. Could be good to add though!
    # 
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # SDXL has two text encoders
        if args.sdxl:
            # import correct text encoder classes
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
               tokenizer_and_encoder_name, args.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                tokenizer_and_encoder_name, subfolder="text_encoder", revision=args.revision
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
            )
            if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
                text_encoders = [text_encoder_two]
                tokenizers = [tokenizer_two]
            else:
                text_encoders = [text_encoder_one, text_encoder_two]
                tokenizers = [tokenizer_one, tokenizer_two]
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )
        # Can custom-select VAE (used in original SDXL tuning)
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision
        )
        # clone of model
        ref_unet = UNet2DConditionModel.from_pretrained(
            args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
            subfolder="unet", revision=args.revision
        )
    if args.unet_init:
        print("Initializing unet from", args.unet_init)
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_init if args.unet_init else args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Conditional adapter for SD1.5 (csft/cdpo)
    cond_adapter = None
    if (not args.sdxl) and (args.train_method in ["csft", "cdpo"]):
        try:
            hidden_size = getattr(text_encoder.config, "hidden_size", 768)
        except Exception:
            hidden_size = 768
        cond_adapter = SD15ConditionAdapter(
            hidden_size=hidden_size,
            projector_type=args.cond_projector_type,
            mlp_hidden_dim=args.cond_mlp_hidden_dim,
            num_condition_tokens=args.cond_num_tokens,
        )

        # Optionally initialize from a pretrained conditional adapter on disk
        if args.cond_adapter_init:
            load_dir = args.cond_adapter_init
            candidate = os.path.join(load_dir, "cond_adapter")
            if os.path.isdir(candidate):
                load_dir = candidate
            try:
                loaded_adapter = SD15ConditionAdapter.from_pretrained(load_dir)
                cond_adapter = loaded_adapter
                print(f"Initialized conditional adapter from '{load_dir}'")
            except Exception as e:
                logger.warning(f"Failed to initialize conditional adapter from '{load_dir}': {e}")

    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    # Optionally freeze UNet for CSFT when training only conditional adapter
    if args.csft_cond_only:
        unet.requires_grad_(False)

    if args.ip_adapter:
        adapter_modules = init_adapter(unet)
        unet = IPAdapter(unet, cond_adapter if not hasattr(cond_adapter, "module") else cond_adapter.module, adapter_modules, args.ip_adapter_ckpt)

    if args.train_method in ['dpo', 'cdpo']:
        if args.train_method == "cdpo":
            import copy
            ref_unet = copy.deepcopy(unet)
        ref_unet.requires_grad_(False)    

    if args.class_conditioning:
        if args.train_method not in ["csft", "cdpo"]:
            raise ValueError("class_conditioning is currently supported only for csft/cdpo training methods")
        if args.simultaneous_conditioning:
            raise ValueError("class_conditioning is not compatible with simultaneous conditioning")
        if args.jeremy_conditioning:
            raise ValueError("class_conditioning is not compatible with jeremy conditioning")
        try:
            class_embed_dim = unet.time_embedding.linear_1.out_features
        except AttributeError:
            class_embed_dim = getattr(unet.config, "time_embed_dim", None)
        if class_embed_dim is None:
            raise ValueError("Unable to determine UNet time embedding dimension for class conditioning")
        class_embed_layer = torch.nn.Embedding(2, class_embed_dim)
        torch.nn.init.zeros_(class_embed_layer.weight)
    else:
        class_embed_layer = None

    # xformers efficient attention
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        if not args.ip_adapter:
            unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    # BRAM NOTE: We're using >=0.16.0. Below was a bit of a bug hive. I hacked around it, but ideally ref_unet wouldn't
    # be getting passed here
    # 
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            # Save UNet and optional conditional adapter. Unwrap to handle DDP/FSDP wrappers.
            for model in list(models):
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, UNet2DConditionModel):
                    unwrapped.save_pretrained(os.path.join(output_dir, "unet"))
                elif isinstance(unwrapped, SD15ConditionAdapter):
                    unwrapped.save_pretrained(os.path.join(output_dir, "cond_adapter"))
                elif isinstance(unwrapped, IPAdapter):
                    unwrapped.save_pretrained(os.path.join(output_dir, "ip_adapter"), cond_only=args.csft_cond_only)
                if len(weights) > 0:
                    weights.pop()

        def load_model_hook(models, input_dir):
            # Pop models to signal we've handled loading. Unwrap to access real modules.
            for _ in range(len(models)):
                model = models.pop()
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    unwrapped.register_to_config(**load_model.config)
                    unwrapped.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(unwrapped, SD15ConditionAdapter):
                    try:
                        load_adapter = SD15ConditionAdapter.from_pretrained(os.path.join(input_dir, "cond_adapter"))
                        unwrapped.load_state_dict(load_adapter.state_dict())
                    except Exception:
                        pass
                elif isinstance(unwrapped, IPAdapter):
                    # load_adapter = IPAdapter.load_from_checkpoint(os.path.join(input_dir, "ip_adapter"))
                    # unwrapped.load_state_dict(load_adapter.state_dict())
                    # del load_adapter
                    print("Loading IPAdapter should be conducted with ip_adapter_ckpt")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl: #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
        print("Enabling gradient checkpointing, either because you asked for this or because you're using SDXL")
        unet.enable_gradient_checkpointing()

    # Bram Note: haven't touched
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Build optimizer parameters (optionally exclude UNet if csft_cond_only)
    optim_params = []
    if not args.csft_cond_only:
        optim_params += list(unet.parameters())
    else:
        # When using IP-Adapter, we train the adapter on UNet; avoid also including
        # the separate conditional adapter params which are unused in this mode.
        if (cond_adapter is not None) and (not args.ip_adapter):
            optim_params += list(cond_adapter.parameters())
        if args.ip_adapter and hasattr(unet, "adapter_modules"):
            for p in unet.adapter_modules.parameters():
                p.requires_grad = True
            for p in unet.image_proj_model.parameters():
                p.requires_grad = True
            optim_params += list(unet.adapter_modules.parameters()) + list(unet.image_proj_model.parameters())


    if class_embed_layer is not None:
        optim_params += list(class_embed_layer.parameters())

    if args.use_adafactor or args.sdxl:
        print("Using Adafactor either because you asked for it or you're using SDXL")
        optimizer = transformers.Adafactor(optim_params,
                                           lr=args.learning_rate,
                                           weight_decay=args.adam_weight_decay,
                                           clip_threshold=1.0,
                                           scale_parameter=False,
                                          relative_step=False)
    else:
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        
        
        
    # Load scores mapping if provided
    scores_mapping = None
    if args.scores_mapping_file is not None:
        logger.info(f"Loading scores mapping from {args.scores_mapping_file}")
        import json
        import pickle
        import hashlib
        if args.scores_mapping_file.endswith(('.pkl', '.pickle')):
            with open(args.scores_mapping_file, 'rb') as f:
                scores_mapping = pickle.load(f)
        else:
            with open(args.scores_mapping_file, 'r') as f:
                scores_mapping = json.load(f)
        logger.info(f"Loaded scores mapping with {len(scores_mapping)} entries")
    
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.streaming:
            logger.info("Loading dataset in streaming mode (no disk download)")
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            # data_dir=args.train_data_dir,
            streaming=args.streaming,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files[args.split] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Helper function to get scores from mapping (used in collate_fn)
    def get_scores_from_mapping(example, idx=None):
        """Get scores for an example from the scores mapping."""
        if scores_mapping is None:
            return None
        
        # Try to get scores by key if it exists in the example
        if 'key' in example and example['key'] in scores_mapping:
            return scores_mapping[example['key']]
        # Try using __key__ (WebDataset format)
        elif '__key__' in example and example['__key__'] in scores_mapping:
            return scores_mapping[example['__key__']]
        # Otherwise try using index as key (for datasets where samples are in order)
        elif idx is not None and str(idx) in scores_mapping:
            return scores_mapping[str(idx)]
        else:
            # Debug: Print first few misses to diagnose key mismatch
            if idx is not None and idx < 3:
                key_tried = example.get('key', 'N/A')
                # Show a few example keys from the mapping for comparison
                sample_keys = list(scores_mapping.keys())[:5] if scores_mapping else []
                logger.warning(f"Score lookup FAILED for idx={idx}, key='{key_tried}'. Example mapping keys: {sample_keys}")
            # Default to NaN if not found
            return {
                "pickscore": [float('nan'), float('nan')],
                "aesthetic": [float('nan'), float('nan')],
                "clip_score": [float('nan'), float('nan')],
                "hps_score": [float('nan'), float('nan')],
            }
    
    # Note: Scores will be added on-the-fly in collate_fn (no preprocessing needed)
    if scores_mapping is not None:
        logger.info(f"Scores mapping loaded with {len(scores_mapping)} entries")
        logger.info("Scores will be added on-the-fly during training (no preprocessing step)")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if args.streaming:
        # For streaming datasets, peek at first example to get column names
        first_example = next(iter(dataset[args.split]))
        column_names = list(first_example.keys())
        logger.info(f"Streaming dataset columns: {column_names}")
        
        # For WebDataset format, map raw column names to expected names
        if 'jpg_0.jpg' in column_names or '__key__' in column_names:
            logger.info("Detected WebDataset format in streaming mode - mapping column names")
            # Map WebDataset column names to standard names
            column_name_mapping = {}
            for col in column_names:
                if col == 'jpg_0.jpg':
                    column_name_mapping['jpg_0'] = 'jpg_0.jpg'
                elif col == 'jpg_1.jpg':
                    column_name_mapping['jpg_1'] = 'jpg_1.jpg'
                elif col == 'label_0.txt':
                    column_name_mapping['label_0'] = 'label_0.txt'
                elif col == 'label_1.txt':
                    column_name_mapping['label_1'] = 'label_1.txt'
                elif col == 'original_prompt.txt':
                    column_name_mapping['caption'] = 'original_prompt.txt'
                elif col == '__key__':
                    column_name_mapping['key'] = '__key__'
            
            logger.info(f"Column mapping: {column_name_mapping}")
            # For WebDataset, we'll use the mapped names
            caption_column = 'original_prompt.txt'
        else:
            caption_column = None
    else:
        column_names = dataset[args.split].column_names
        caption_column = None

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    # Pairwise or conditional datasets don't require a single image_column. This includes Pick-a-Pic and MVV Full.
    if (args.dataset_name and (('pickapic' in args.dataset_name) or ('mvv_full' in args.dataset_name))) or (args.train_method in ['dpo', 'cdpo', 'csft']):
        # Pairwise or conditional modes don't require a single image_column
        pass
    elif args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    # Handle caption column for both streaming and non-streaming
    if caption_column is None:
        if args.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        # For WebDataset streaming, use 'caption' after mapping, otherwise use caption_column
        cap_col = 'caption' if (args.streaming and 'caption' in examples) else caption_column
        for caption in examples[cap_col]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    
    ##### START BIG OLD DATASET BLOCK #####
    
    # Check if dataset has all_scores fields (either from dataset name or from scores_mapping_file)
    has_all_scores = ("all_scores" in args.dataset_name or 
                      (scores_mapping is not None) or
                      ('pickscore' in column_names and 'aesthetic' in column_names and 
                       'clip_score' in column_names and 'hps_score' in column_names))
    
    if has_all_scores:
        logger.info("Using all_scores mode: pickscore, aesthetic, clip_score, hps_score")
    else:
        logger.info("Using mvv_full mode: mps_probs, vqa_scores, vila_scores")
    
    #### START PREPROCESSING/COLLATION ####
    if args.train_method in ['dpo', 'cdpo']:
        print("Ignoring image_column variable, reading from jpg_0 and jpg_1")
        def preprocess_train(examples):
            # Handle WebDataset format column names in streaming mode
            if args.streaming and 'jpg_0.jpg' in examples:
                # Map WebDataset columns to expected names
                # Use 'caption' as the target key regardless of source
                
                # Robustly derive label from multiple possible encodings (reversed from build_all_scores_hf_dataset.py)
                # When jpg_0 wins (l0 > l1), label_0 = 1
                def decode_label(sample_idx):
                    l0_raw = examples.get('label_0.txt', examples.get('label_0', [None] * len(examples.get('jpg_0.jpg', []))))[sample_idx]
                    l1_raw = examples.get('label_1.txt', examples.get('label_1', [None] * len(examples.get('jpg_0.jpg', []))))[sample_idx]
                    
                    def decode_numeric(value):
                        if isinstance(value, (bytes, bytearray)):
                            try:
                                s = value.decode('utf-8', errors='ignore').strip().strip('"')
                            except Exception:
                                return None
                        elif isinstance(value, str):
                            s = value.strip().strip('"')
                        elif isinstance(value, (int, float, np.floating, np.integer)):
                            return float(value)
                        else:
                            return None
                        try:
                            return float(s)
                        except Exception:
                            return None
                    
                    l0 = decode_numeric(l0_raw)
                    l1 = decode_numeric(l1_raw)
                    
                    label_value = None
                    if l0 is not None and l1 is not None:
                        # REVERSED: if l0 > l1, jpg_0 wins, so label_0 = 1
                        if l0 > l1:
                            label_value = 1
                        elif l1 > l0:
                            label_value = 0
                    elif l0 is not None:
                        # REVERSED: if l0 >= 0.5, jpg_0 wins, so label_0 = 1
                        if l0 >= 0.5:
                            label_value = 1
                        else:
                            label_value = 0
                    elif l1 is not None:
                        # REVERSED: if l1 >= 0.5, jpg_1 wins, so label_0 = 0
                        if l1 >= 0.5:
                            label_value = 0
                        else:
                            label_value = 1
                    
                    if label_value is None:
                        label_value = -1  # Invalid/tie
                    
                    return label_value
                
                num_samples = len(examples.get('jpg_0.jpg', examples.get('jpg_0', [])))
                labels = [decode_label(i) for i in range(num_samples)]
                
                # Debug: check what fields are available for unique identification
                # print(f"DEBUG: Available fields: {list(examples.keys())}")
                # print(f"DEBUG: Has __url__: {'__url__' in examples}, Has __key__: {'__key__' in examples}")
                # if '__url__' in examples:
                #     print(f"DEBUG: __url__ sample: {examples['__url__'][:2] if len(examples.get('__url__', [])) >= 2 else examples.get('__url__')}")
                # if '__key__' in examples:
                #     print(f"DEBUG: __key__ sample: {examples['__key__'][:2] if len(examples.get('__key__', [])) >= 2 else examples.get('__key__')}")
                
                examples_mapped = {
                    'jpg_0': examples.get('jpg_0.jpg', examples.get('jpg_0')),
                    'jpg_1': examples.get('jpg_1.jpg', examples.get('jpg_1')),
                    'label_0': labels,
                    'caption': [x.decode('utf-8') if isinstance(x, bytes) else x for x in examples.get('original_prompt.txt', examples.get('caption', []))],
                }
                
                # __key__ alone is not unique (resets per shard), need to construct unique ID
                # Combine shard ID (from __url__) with __key__ to create globally unique keys
                if '__key__' in examples and '__url__' in examples:
                    # Extract shard ID from URL (e.g., "000001" from "path/to/000001.tar")
                    urls = examples['__url__']
                    keys = examples['__key__']
                    
                    unique_keys = []
                    for url, key in zip(urls, keys):
                        # Extract filename from URL and get shard ID
                        filename = os.path.basename(url) if isinstance(url, str) else url
                        shard_id = os.path.splitext(filename)[0] if isinstance(filename, str) else str(filename)
                        # Create unique key: shard_id + "_" + local_key
                        unique_key = f"{shard_id}_{key}"
                        unique_keys.append(unique_key)
                    
                    # print(f"DEBUG: Constructed unique_keys sample: {unique_keys[:3]}")
                    examples_mapped['key'] = unique_keys
                    examples_mapped['url'] = urls
                elif '__key__' in examples:
                    # Fallback: use __key__ alone (will have collisions but better than nothing)
                    logger.warning("__url__ not found in streaming data, using non-unique __key__ (may cause score lookup issues)")
                    raise ValueError("__url__ not found in streaming data")
                    
                examples = examples_mapped
                # For tokenization, always use 'caption' after mapping
                examples['caption'] = examples.get('caption', [])
                
                # Print sample to diagnose key uniqueness issue
                # if len(examples['label_0']) > 0:
                #     print(f"  Sample 0: key={examples.get('key', ['N/A'])[0]}, url={examples.get('url', ['N/A'])[0]}, caption={examples['caption'][0][:60]}...")
            all_pixel_values = []
            for col_name in ['jpg_0', 'jpg_1']:
                # Handle both bytes (non-streaming) and PIL Images (streaming)
                images = []
                for im_data in examples[col_name]:
                    if isinstance(im_data, bytes):
                        # Non-streaming: decode from bytes
                        images.append(Image.open(io.BytesIO(im_data)).convert("RGB"))
                    else:
                        # Streaming: already a PIL Image
                        images.append(im_data.convert("RGB") if hasattr(im_data, 'convert') else im_data)
                pixel_values = [train_transforms(image) for image in images]
                all_pixel_values.append(pixel_values)
            # Double on channel dim, jpg_y then jpg_w
            im_tup_iterator = zip(*all_pixel_values)
            combined_pixel_values = []
            if has_all_scores:
                win_pickscore = []
                lose_pickscore = []
                win_aesthetic = []
                lose_aesthetic = []
                win_clip_score = []
                lose_clip_score = []
                win_hps_score = []
                lose_hps_score = []
                
                # Get scores from mapping if not in examples
                if 'pickscore' not in examples and scores_mapping is not None:
                    pickscores_list = []
                    aesthetics_list = []
                    clip_scores_list = []
                    hps_scores_list = []
                    for i in range(len(examples['label_0'])):
                        example_dict = {k: v[i] if isinstance(v, list) else v for k, v in examples.items()}
                        scores = get_scores_from_mapping(example_dict, idx=i)
                        pickscores_list.append(scores['pickscore'])
                        aesthetics_list.append(scores['aesthetic'])
                        clip_scores_list.append(scores['clip_score'])
                        hps_scores_list.append(scores['hps_score'])
                    examples['pickscore'] = pickscores_list
                    examples['aesthetic'] = aesthetics_list
                    examples['clip_score'] = clip_scores_list
                    examples['hps_score'] = hps_scores_list
                
                for im_tup, label_0, pickscore, aesthetic, clip_score, hps_score in zip(im_tup_iterator, examples['label_0'], examples['pickscore'], examples['aesthetic'], examples['clip_score'], examples['hps_score']):
                    if label_0==0 and (not args.choice_model): # don't want to flip things if using choice_model for AI feedback
                        im_tup = im_tup[::-1]
                    combined_im = torch.cat(im_tup, dim=0) # no batch dim
                    combined_pixel_values.append(combined_im)
                    win_idx = 0 if label_0==1 else 1
                    win_pickscore.append(normalize_pick(pickscore[win_idx]))
                    lose_pickscore.append(normalize_pick(pickscore[1-win_idx]))
                    win_aesthetic.append(normalize_aes(aesthetic[win_idx]))
                    lose_aesthetic.append(normalize_aes(aesthetic[1-win_idx]))
                    win_clip_score.append(normalize_clip(clip_score[win_idx]))
                    lose_clip_score.append(normalize_clip(clip_score[1-win_idx]))
                    win_hps_score.append(normalize_hps(hps_score[win_idx]))
                    lose_hps_score.append(normalize_hps(hps_score[1-win_idx]))
                examples["win_pickscore"] = win_pickscore
                examples["lose_pickscore"] = lose_pickscore
                examples["win_aesthetic"] = win_aesthetic
                examples["lose_aesthetic"] = lose_aesthetic
                examples["win_clip_score"] = win_clip_score
                examples["lose_clip_score"] = lose_clip_score
                examples["win_hps_score"] = win_hps_score
                examples["lose_hps_score"] = lose_hps_score
                examples["pixel_values"] = combined_pixel_values
            else:
                win_mps = []
                lose_mps = []
                win_vqa = []
                lose_vqa = []
                win_vila = []
                lose_vila = []
                for im_tup, label_0, mps_probs, vqa_scores, vila_scores in zip(im_tup_iterator, examples['label_0'], examples['mps_scores'], examples['vqa_scores'], examples['vila_scores']):
                    if label_0==0 and (not args.choice_model): # don't want to flip things if using choice_model for AI feedback
                        im_tup = im_tup[::-1]
                    combined_im = torch.cat(im_tup, dim=0) # no batch dim
                    combined_pixel_values.append(combined_im)
                    win_idx = 0 if label_0==1 else 1
                    win_mps.append(normalize_mps(mps_probs[win_idx]))
                    lose_mps.append(normalize_mps(mps_probs[1-win_idx]))
                    win_vqa.append(normalize_vqa(vqa_scores[win_idx]))
                    lose_vqa.append(normalize_vqa(vqa_scores[1-win_idx]))
                    win_vila.append(normalize_vila(vila_scores[win_idx]))
                    lose_vila.append(normalize_vila(vila_scores[1-win_idx]))
                examples["win_mps_probs"] = win_mps
                examples["lose_mps_probs"] = lose_mps
                examples["win_vqa_scores"] = win_vqa
                examples["lose_vqa_scores"] = lose_vqa
                examples["win_vila_scores"] = win_vila
                examples["lose_vila_scores"] = lose_vila
                examples["pixel_values"] = combined_pixel_values
            # SDXL takes raw prompts
            if not args.sdxl: examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d =  {"pixel_values": pixel_values}
            # SDXL takes raw prompts
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            if args.train_method == 'cdpo':
                if args.multi_dim:
                    def cond_text(sample1, sample2):
                        if sample1 > sample2:
                            return 'win'
                        elif sample1 < sample2:
                            return 'lose'
                        else:
                            return 'tie'
                    if has_all_scores:
                        win_conds = [f'win {cond_text(example["win_pickscore"], example["lose_pickscore"])} {cond_text(example["win_aesthetic"], example["lose_aesthetic"])} {cond_text(example["win_clip_score"], example["lose_clip_score"])} {cond_text(example["win_hps_score"], example["lose_hps_score"])}' for example in examples]
                        lose_conds = [f'lose {cond_text(example["lose_pickscore"], example["win_pickscore"])} {cond_text(example["lose_aesthetic"], example["win_aesthetic"])} {cond_text(example["lose_clip_score"], example["win_clip_score"])} {cond_text(example["lose_hps_score"], example["win_hps_score"])}' for example in examples]
                    else:
                        win_conds = [f'win {cond_text(example["win_mps_probs"], example["lose_mps_probs"])} {cond_text(example["win_vqa_scores"], example["lose_vqa_scores"])} {cond_text(example["win_vila_scores"], example["lose_vila_scores"])}' for example in examples]
                        lose_conds = [f'lose {cond_text(example["lose_mps_probs"], example["win_mps_probs"])} {cond_text(example["lose_vqa_scores"], example["win_vqa_scores"])} {cond_text(example["lose_vila_scores"], example["win_vila_scores"])}' for example in examples]
                    # conds = [f'win {cond_text(example["win_vqa_scores"], example["lose_vqa_scores"])} {cond_text(example["win_vila_scores"], example["lose_vila_scores"])}' for example in examples] + [f'lose {cond_text(example["lose_vqa_scores"], example["win_vqa_scores"])} {cond_text(example["lose_vila_scores"], example["win_vila_scores"])}' for example in examples]
                    # print(win_conds[0], lose_conds[0])
                    # Tokenize win and lose conditions separately, then stack along dimension 1
                    win_cond_input_ids = tokenizer(win_conds, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
                    lose_cond_input_ids = tokenizer(lose_conds, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
                    # Stack to shape [batch_size, 2, seq_len] where dim 1 is [win, lose]
                    cond_input_ids = torch.stack([win_cond_input_ids, lose_cond_input_ids], dim=1)
                    # For multi_dim, store which condition was used (always 0 for win in this case since we store both)
                    # This is used for non-simultaneous conditioning to select which condition to use
                    cond_is_positive = torch.ones(len(examples), dtype=torch.long)  # Placeholder, actual selection happens in training loop
                    # print(f"DEBUG: key: {examples[0]['key']}, win_conds: {win_conds[0]}, lose_conds: {lose_conds[0]}")
                else:
                    conds = []
                    for _ in examples:
                        conds.append(args.cond_positive_text if random.random() < 0.5 else args.cond_negative_text)
                    # For non-multi_dim, just tokenize - no extra dimension needed
                    cond_input_ids = tokenizer(conds, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
                    # Store condition flags as tensor: 1 for positive/win text, 0 for negative/lose text
                    cond_is_positive = torch.tensor([1 if c == args.cond_positive_text else 0 for c in conds], dtype=torch.long)
                return_d["cond_input_ids"] = cond_input_ids
                return_d["cond_is_positive"] = cond_is_positive
                
            if args.choice_model:
                # If using AIF then deliver image data for choice model to determine if should flip pixel values
                for k in ['jpg_0', 'jpg_1']:
                    return_d[k] = [Image.open(io.BytesIO( example[k])).convert("RGB")
                                   for example in examples]
                return_d["caption"] = [example["caption"] for example in examples] 
            return return_d
         
        if args.choice_model:
            # TODO: Fancy way of doing this?
            if args.choice_model == 'hps':
                from utils.hps_utils import Selector
            elif args.choice_model == 'clip':
                from utils.clip_utils import Selector
            elif args.choice_model == 'pickscore':
                from utils.pickscore_utils import Selector
            elif args.choice_model == 'aes':
                from utils.aes_utils import Selector
            selector = Selector('cpu' if args.sdxl else accelerator.device)

            def do_flip(jpg0, jpg1, prompt):
                scores = selector.score([jpg0, jpg1], prompt)
                return scores[1] > scores[0]
            def choice_model_says_flip(batch):
                assert len(batch['caption'])==1 # Can switch to iteration but not needed for nwo
                return do_flip(batch['jpg_0'][0], batch['jpg_1'][0], batch['caption'][0])
    elif args.train_method == 'sft':
        def preprocess_train(examples):
            if 'pickapic' in args.dataset_name or 'mvv_full' in args.dataset_name:
                images = []
                # Probably cleaner way to do this iteration
                for im_0_bytes, im_1_bytes, label_0 in zip(examples['jpg_0'], examples['jpg_1'], examples['label_0']):
                    assert label_0 in (0, 1)
                    im_bytes = im_0_bytes if label_0==1 else im_1_bytes
                    images.append(Image.open(io.BytesIO(im_bytes)).convert("RGB"))
            else:
                images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            if not args.sdxl: examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d =  {"pixel_values": pixel_values}
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d
    elif args.train_method == 'csft':
        def preprocess_train(examples):
            # Handle WebDataset format column names in streaming mode
            if args.streaming and 'jpg_0.jpg' in examples:
                # Map WebDataset columns to expected names
                examples_mapped = {
                    'jpg_0': examples.get('jpg_0.jpg', examples.get('jpg_0')),
                    'jpg_1': examples.get('jpg_1.jpg', examples.get('jpg_1')),
                    'label_0': [int(float(x.decode('utf-8') if isinstance(x, bytes) else x)) for x in examples.get('label_0.txt', examples.get('label_0', []))],
                    'caption': [x.decode('utf-8') if isinstance(x, bytes) else x for x in examples.get('original_prompt.txt', examples.get('caption', []))],
                }
                
                # Construct unique keys (same as dpo branch)
                if '__key__' in examples and '__url__' in examples:
                    # Extract shard ID from URL and combine with local key
                    urls = examples['__url__']
                    keys = examples['__key__']
                    
                    unique_keys = []
                    for url, key in zip(urls, keys):
                        filename = os.path.basename(url) if isinstance(url, str) else url
                        shard_id = os.path.splitext(filename)[0] if isinstance(filename, str) else str(filename)
                        unique_key = f"{shard_id}_{key}"
                        unique_keys.append(unique_key)
                    
                    examples_mapped['key'] = unique_keys
                    examples_mapped['url'] = urls
                elif '__key__' in examples:
                    logger.warning("__url__ not found in streaming data for csft, using non-unique __key__")
                    raise ValueError("__url__ not found in streaming data")
                
                examples = examples_mapped
            
            if has_all_scores:
                win_images = []
                lose_images = []
                captions = []
                win_pickscore = []
                lose_pickscore = []
                win_aesthetic = []
                lose_aesthetic = []
                win_clip_score = []
                lose_clip_score = []
                win_hps_score = []
                lose_hps_score = []
                
                # Get scores from mapping if not in examples
                if 'pickscore' not in examples and scores_mapping is not None:
                    pickscores_list = []
                    aesthetics_list = []
                    clip_scores_list = []
                    hps_scores_list = []
                    for i in range(len(examples['label_0'])):
                        example_dict = {k: v[i] if isinstance(v, list) else v for k, v in examples.items()}
                        scores = get_scores_from_mapping(example_dict, idx=i)
                        pickscores_list.append(scores['pickscore'])
                        aesthetics_list.append(scores['aesthetic'])
                        clip_scores_list.append(scores['clip_score'])
                        hps_scores_list.append(scores['hps_score'])
                    examples['pickscore'] = pickscores_list
                    examples['aesthetic'] = aesthetics_list
                    examples['clip_score'] = clip_scores_list
                    examples['hps_score'] = hps_scores_list
                
                for im_0_data, im_1_data, label_0, cap, pickscore, aesthetic, clip_score, hps_score in zip(examples['jpg_0'], examples['jpg_1'], examples['label_0'], examples['caption'], examples['pickscore'], examples['aesthetic'], examples['clip_score'], examples['hps_score']):
                    assert label_0 in (0, 1)
                    im_win_data = im_0_data if label_0==1 else im_1_data
                    im_lose_data = im_1_data if label_0==1 else im_0_data
                    
                    # Handle both bytes (non-streaming) and PIL Images (streaming)
                    if isinstance(im_win_data, bytes):
                        win_images.append(Image.open(io.BytesIO(im_win_data)).convert("RGB"))
                        lose_images.append(Image.open(io.BytesIO(im_lose_data)).convert("RGB"))
                    else:
                        win_images.append(im_win_data.convert("RGB") if hasattr(im_win_data, 'convert') else im_win_data)
                        lose_images.append(im_lose_data.convert("RGB") if hasattr(im_lose_data, 'convert') else im_lose_data)
                    captions.append(cap)
                    win_idx = 0 if label_0==1 else 1
                    win_pickscore.append(normalize_pick(pickscore[win_idx]))
                    lose_pickscore.append(normalize_pick(pickscore[1-win_idx]))
                    win_aesthetic.append(normalize_aes(aesthetic[win_idx]))
                    lose_aesthetic.append(normalize_aes(aesthetic[1-win_idx]))
                    win_clip_score.append(normalize_clip(clip_score[win_idx]))
                    lose_clip_score.append(normalize_clip(clip_score[1-win_idx]))
                    win_hps_score.append(normalize_hps(hps_score[win_idx]))
                    lose_hps_score.append(normalize_hps(hps_score[1-win_idx]))
            elif 'pickapic' in args.dataset_name or 'mvv_full' in args.dataset_name:
                win_images = []
                lose_images = []
                captions = []
                win_mps = []
                lose_mps = []
                win_vqa = []
                lose_vqa = []
                win_vila = []
                lose_vila = []
                for im_0_data, im_1_data, label_0, cap, mps_probs, vqa_scores, vila_scores in zip(examples['jpg_0'], examples['jpg_1'], examples['label_0'], examples['caption'], examples['mps_scores'], examples['vqa_scores'], examples['vila_scores']):
                    assert label_0 in (0, 1)
                    im_win_data = im_0_data if label_0==1 else im_1_data
                    im_lose_data = im_1_data if label_0==1 else im_0_data
                    
                    # Handle both bytes (non-streaming) and PIL Images (streaming)
                    if isinstance(im_win_data, bytes):
                        win_images.append(Image.open(io.BytesIO(im_win_data)).convert("RGB"))
                        lose_images.append(Image.open(io.BytesIO(im_lose_data)).convert("RGB"))
                    else:
                        win_images.append(im_win_data.convert("RGB") if hasattr(im_win_data, 'convert') else im_win_data)
                        lose_images.append(im_lose_data.convert("RGB") if hasattr(im_lose_data, 'convert') else im_lose_data)
                    captions.append(cap)
                    win_idx = 0 if label_0==1 else 1
                    win_mps.append(normalize_mps(mps_probs[win_idx]))
                    lose_mps.append(normalize_mps(mps_probs[1-win_idx]))
                    win_vqa.append(normalize_vqa(vqa_scores[win_idx]))
                    lose_vqa.append(normalize_vqa(vqa_scores[1-win_idx]))
                    win_vila.append(normalize_vila(vila_scores[win_idx]))
                    lose_vila.append(normalize_vila(vila_scores[1-win_idx]))
            else:
                # Fallback: single image datasets, treat image as win and duplicate as lose
                for image, cap in zip(examples[image_column], examples[caption_column]):
                    img = image.convert("RGB")
                    win_images.append(img)
                    lose_images.append(img)
                    captions.append(cap)
            examples["pixel_values_win"] = [train_transforms(img) for img in win_images]
            examples["pixel_values_lose"] = [train_transforms(img) for img in lose_images]
            if not args.sdxl: examples["input_ids"] = tokenize_captions({caption_column: captions})
            else: examples["caption"] = captions
            if args.multi_dim:
                if has_all_scores:
                    examples["win_pickscore"] = win_pickscore
                    examples["lose_pickscore"] = lose_pickscore
                    examples["win_aesthetic"] = win_aesthetic
                    examples["lose_aesthetic"] = lose_aesthetic
                    examples["win_clip_score"] = win_clip_score
                    examples["lose_clip_score"] = lose_clip_score
                    examples["win_hps_score"] = win_hps_score
                    examples["lose_hps_score"] = lose_hps_score
                else:
                    examples["win_mps_probs"] = win_mps
                    examples["lose_mps_probs"] = lose_mps
                    examples["win_vqa_scores"] = win_vqa
                    examples["lose_vqa_scores"] = lose_vqa
                    examples["win_vila_scores"] = win_vila
                    examples["lose_vila_scores"] = lose_vila
            return examples

        def collate_fn(examples):
            win = torch.stack([ex["pixel_values_win"] for ex in examples])
            lose = torch.stack([ex["pixel_values_lose"] for ex in examples])
            pixel_values = torch.cat([win, lose], dim=0)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d = {"pixel_values": pixel_values}
            if args.sdxl:
                caps = [ex["caption"] for ex in examples]
                return_d["caption"] = caps + caps
            else:
                ids = torch.stack([ex["input_ids"] for ex in examples])
                return_d["input_ids"] = torch.cat([ids, ids], dim=0)
            # Provide aligned condition texts: first half positive, second half negative
            if args.multi_dim:
                def cond_text(sample1, sample2):
                    if sample1 > sample2:
                        return 'win'
                    elif sample1 < sample2:
                        return 'lose'
                    else:
                        return 'tie'
                if has_all_scores:
                    conds = [f'win {cond_text(ex["win_pickscore"], ex["lose_pickscore"])} {cond_text(ex["win_aesthetic"], ex["lose_aesthetic"])} {cond_text(ex["win_clip_score"], ex["lose_clip_score"])} {cond_text(ex["win_hps_score"], ex["lose_hps_score"])}' for ex in examples] + [f'lose {cond_text(ex["lose_pickscore"], ex["win_pickscore"])} {cond_text(ex["lose_aesthetic"], ex["win_aesthetic"])} {cond_text(ex["lose_clip_score"], ex["win_clip_score"])} {cond_text(ex["lose_hps_score"], ex["win_hps_score"])}' for ex in examples]
                else:
                    conds = [f'win {cond_text(ex["win_mps_probs"], ex["lose_mps_probs"])} {cond_text(ex["win_vqa_scores"], ex["lose_vqa_scores"])} {cond_text(ex["win_vila_scores"], ex["lose_vila_scores"])}' for ex in examples] + [f'lose {cond_text(ex["lose_mps_probs"], ex["win_mps_probs"])} {cond_text(ex["lose_vqa_scores"], ex["win_vqa_scores"])} {cond_text(ex["lose_vila_scores"], ex["win_vila_scores"])}' for ex in examples]
                # conds = [f'win {cond_text(ex["win_vqa_scores"], ex["lose_vqa_scores"])} {cond_text(ex["win_vila_scores"], ex["lose_vila_scores"])}' for ex in examples] + [f'lose {cond_text(ex["lose_vqa_scores"], ex["win_vqa_scores"])} {cond_text(ex["lose_vila_scores"], ex["win_vila_scores"])}' for ex in examples]
                # print(conds[0], conds[-1])
            else:
                conds = [args.cond_positive_text for _ in examples] + [args.cond_negative_text for _ in examples]
            # Tokenize condition texts to avoid string concatenation issues with Accelerate
            cond_input_ids = tokenizer(conds, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
            return_d["cond_input_ids"] = cond_input_ids
            return return_d
    #### END PREPROCESSING/COLLATION ####
    
    ### DATASET #####
    with accelerator.main_process_first():
        # Drop tie/invalid labels for non-streaming datasets (before preprocessing)
        if not args.streaming:
            if 'label_0' in dataset[args.split].column_names:
                orig_len = dataset[args.split].num_rows
                keep_idx = [i for i, l in enumerate(dataset[args.split]['label_0']) if l in (0, 1)]
                if len(keep_idx) != orig_len:
                    dataset[args.split] = dataset[args.split].select(keep_idx)
                    new_len = dataset[args.split].num_rows
                    print(f"Dropped {orig_len - new_len}/{orig_len} tie/invalid label examples")

        # Normalize semantics for hagiss/mvv_full: label_0==0 means jpg_0 wins
        if 'hagiss/mvv_full' in args.dataset_name and 'label_0' in dataset[args.split].column_names:
            dataset[args.split] = dataset[args.split].map(lambda example: {'label_0': 1 - example['label_0']})

        # if 'pickapic' in args.dataset_name:
            # Below if if want to train on just the Dreamlike vs dreamlike pairs
            if args.dreamlike_pairs_only:
                orig_len = dataset[args.split].num_rows
                dream_like_idx = [i for i,(m0,m1) in enumerate(zip(dataset[args.split]['model_0'],
                                                                   dataset[args.split]['model_1']))
                                  if ( ('dream' in m0) and ('dream' in m1) )]
                dataset[args.split] = dataset[args.split].select(dream_like_idx)
                new_len = dataset[args.split].num_rows
                print(f"Eliminated {orig_len - new_len}/{orig_len} non-dreamlike gens for Pick-a-pic")
                
        if args.max_train_samples is not None:
            if args.streaming:
                # For streaming datasets, use take() instead of select()
                dataset[args.split] = dataset[args.split].shuffle(seed=args.seed, buffer_size=10000).take(args.max_train_samples)
            else:
                dataset[args.split] = dataset[args.split].shuffle(seed=args.seed).select(range(args.max_train_samples))
        
        # Set the training transforms
        if args.streaming:
            # For streaming datasets, use map() instead of with_transform()
            # First apply preprocessing to generate label_0
            train_dataset = dataset[args.split].map(preprocess_train, batched=True, batch_size=args.train_batch_size)
            # Then filter out tie/invalid labels (must happen after preprocessing for WebDataset format)
            logger.info("Filtering out tie/invalid labels (label_0 not in {0, 1}) for streaming dataset")
            train_dataset = train_dataset.filter(lambda example: example.get('label_0', -1) in (0, 1))
        else:
            train_dataset = dataset[args.split].with_transform(preprocess_train)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(args.split=='train' and not args.streaming),  # Don't shuffle streaming datasets
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )
    ##### END BIG OLD DATASET BLOCK #####
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    
    if args.streaming:
        # For streaming datasets, we can't get the length
        # User must specify max_train_steps
        if args.max_train_steps is None:
            raise ValueError(
                "When using --streaming, you must specify --max_train_steps "
                "since streaming datasets don't have a known length."
            )
        num_update_steps_per_epoch = args.max_train_steps  # Dummy value for streaming
        logger.info(f"Streaming mode: using max_train_steps={args.max_train_steps}")
    else:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    
    #### START ACCELERATOR PREP ####
    if cond_adapter is not None and not args.ip_adapter:
        unet, cond_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, cond_adapter, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

        
    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.sdxl:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        print("offload vae (this actually stays as CPU)")
        vae = accelerate.cpu_offload(vae)
        print("Offloading text encoders to cpu")
        text_encoder_one = accelerate.cpu_offload(text_encoder_one)
        text_encoder_two = accelerate.cpu_offload(text_encoder_two)
        if args.train_method in ['dpo', 'cdpo']:
            ref_unet.to(accelerator.device, dtype=weight_dtype)
            print("offload ref_unet")
            ref_unet = accelerate.cpu_offload(ref_unet)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if args.train_method in ['dpo', 'cdpo']:
            ref_unet.to(accelerator.device, dtype=weight_dtype)
        if cond_adapter is not None:
            cond_adapter.to(accelerator.device)
    ### END ACCELERATOR PREP ###
    
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if not args.streaming:
        # Only recalculate for non-streaming datasets
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    else:
        # For streaming, we already set max_train_steps and can't calculate epochs
        logger.info(f"Streaming mode: training for {args.max_train_steps} steps")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Qualitative sampling config
    SAMPLE_EVERY_STEPS = 100
    SAMPLE_INFERENCE_STEPS = 50
    SAMPLE_GUIDANCE_SCALE = 7.5
    SAMPLE_GUIDANCE_SCALE2 = 4
    sample_pipe = None
    sample_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    def _build_or_update_sample_pipeline():
        nonlocal sample_pipe
        unet_infer = accelerator.unwrap_model(unet)
        unet_dtype = next(unet_infer.parameters()).dtype
        if sample_pipe is None:
            if args.sdxl:
                sample_pipe_local = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    revision=args.revision,
                    torch_dtype=unet_dtype,
                )
            else:
                sample_pipe_local = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    revision=args.revision,
                    torch_dtype=unet_dtype,
                )
                # Apply conditional monkey patch for SD1.5 during training sampling
                if (not args.sdxl) and (cond_adapter is not None) and (args.train_method in ["csft", "cdpo"]):
                    try:
                        if args.ip_adapter:
                            monkey_patch_sd15_pipeline_for_ipadapter(sample_pipe_local)
                        else:
                            monkey_patch_sd15_pipeline_for_condition(
                                sample_pipe_local,
                                cond_adapter if not hasattr(cond_adapter, "module") else cond_adapter.module,
                                positive_condition=args.cond_positive_text,
                                negative_condition=args.cond_negative_text,
                            )
                    except Exception as e:
                        logger.warning(f"Conditional CFG monkey-patch failed; proceeding to sample without conditional tokens. Error: {e}")
            # swap in current training UNet
            sample_pipe_local.unet = unet_infer
            try:
                sample_pipe_local.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            sample_pipe_local.to(accelerator.device, torch_dtype=unet_dtype)
            sample_pipe_local.set_progress_bar_config(disable=True)
            sample_pipe = sample_pipe_local
        else:
            sample_pipe.unet = unet_infer
            # keep pipeline modules in sync with UNet dtype
            sample_pipe.to(accelerator.device, torch_dtype=unet_dtype)

    # Generate an initial qualitative sample before training starts (main process only)
    if accelerator.is_main_process:
        _build_or_update_sample_pipeline()
        # Log a quick weight checksum to verify training updates over time
        try:
            _unet_chk = next(accelerator.unwrap_model(unet).parameters()).detach().float()
            accelerator.log({"unet_first_weight_mean": _unet_chk.mean().item()}, step=0)
        except Exception:
            pass
        with torch.inference_mode():
            _gen = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            if args.sdxl:
                images = sample_pipe(
                    prompt=SAMPLE_PROMPTS,
                    num_inference_steps=SAMPLE_INFERENCE_STEPS,
                    guidance_scale=SAMPLE_GUIDANCE_SCALE,
                    generator=_gen,
                ).images
            else:
                try:
                    if args.ip_adapter:
                        images = sample_pipe.__call__(
                            self=sample_pipe,
                            prompt=SAMPLE_PROMPTS,
                            num_inference_steps=SAMPLE_INFERENCE_STEPS,
                            guidance_scale=SAMPLE_GUIDANCE_SCALE,
                            generator=_gen,
                            positive_condition=args.cond_positive_text,
                            negative_condition=args.cond_negative_text,
                        ).images
                        images2 = sample_pipe.__call__(
                            self=sample_pipe,
                            prompt=SAMPLE_PROMPTS,
                            num_inference_steps=SAMPLE_INFERENCE_STEPS,
                            guidance_scale=SAMPLE_GUIDANCE_SCALE2,
                            generator=_gen,
                            positive_condition=args.cond_positive_text,
                            negative_condition=args.cond_negative_text,
                        ).images
                        images3 = sample_pipe.__call__(
                            self=sample_pipe,
                            prompt=SAMPLE_PROMPTS,
                            num_inference_steps=SAMPLE_INFERENCE_STEPS,
                            guidance_scale=1.0,
                            generator=_gen,
                            positive_condition=args.cond_positive_text,
                            negative_condition=args.cond_negative_text,
                        ).images
                    else:
                        images = sample_pipe(
                            prompt=SAMPLE_PROMPTS,
                            num_inference_steps=SAMPLE_INFERENCE_STEPS,
                            guidance_scale=SAMPLE_GUIDANCE_SCALE,
                            generator=_gen,
                        ).images
                        images2 = sample_pipe(
                            prompt=SAMPLE_PROMPTS,
                            num_inference_steps=SAMPLE_INFERENCE_STEPS,
                            guidance_scale=SAMPLE_GUIDANCE_SCALE2,
                            generator=_gen,
                        ).images
                        images3 = sample_pipe(
                            prompt=SAMPLE_PROMPTS,
                            num_inference_steps=SAMPLE_INFERENCE_STEPS,
                            guidance_scale=1.0,
                            generator=_gen,
                        ).images
                except Exception as e:
                    print(f"Failed to sample images: {e}")
                    breakpoint()

            
            # Log list of images to trackers (e.g., W&B)
            try:
                if is_wandb_available():
                    wandb_images = [wandb.Image(img) for img in images]
                    accelerator.log({"samples": wandb_images}, step=0)
                    wandb_images2 = [wandb.Image(img) for img in images2]
                    accelerator.log({"samples2": wandb_images2}, step=0)
                    wandb_images3 = [wandb.Image(img) for img in images3]
                    accelerator.log({"cfg_1": wandb_images3}, step=0)
                else:
                    out_dir = os.path.join(args.output_dir, "samples")
                    os.makedirs(out_dir, exist_ok=True)
                    for idx, img in enumerate(images):
                        img.save(os.path.join(out_dir, f"step000000_{idx}.png"))
            except Exception as e:
                logger.warning(f"Failed to log pre-training samples: {e}")

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

        
    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and (not args.hard_skip_resume):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Dummy processing step {step}, will start training at {resume_step}")
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.train_method in ['dpo', 'cdpo']:
                    # y_w and y_l were concatenated along channel dimension
                    winners, losers = batch["pixel_values"].chunk(2, dim=1)
                    # If using AIF/choice model, determine winners/losers by the selector
                    if args.choice_model:
                        if choice_model_says_flip(batch):
                            winners, losers = losers, winners
                    if args.train_method == 'dpo':
                        feed_pixel_values = torch.cat([winners, losers], dim=0)
                    else:
                        if args.simultaneous_conditioning:
                            feed_pixel_values = torch.cat([winners, losers, losers, winners], dim=0)
                            if args.multi_dim:
                                # cond_input_ids has shape [batch_size, 2, seq_len] where dim 1 is [win, lose]
                                win_cond_input_ids = batch["cond_input_ids"][:, 0, :]  # [batch_size, seq_len]
                                lose_cond_input_ids = batch["cond_input_ids"][:, 1, :]  # [batch_size, seq_len]
                                # Concatenate: [win, lose, win, lose] to align with [winners, losers, losers, winners]
                                batch["cond_input_ids"] = torch.cat([win_cond_input_ids, lose_cond_input_ids, win_cond_input_ids, lose_cond_input_ids], dim=0)
                            else:
                                cond_texts_updated = [args.cond_positive_text for _ in range(winners.shape[0])] + [args.cond_negative_text for _ in range(losers.shape[0])] + [args.cond_positive_text for _ in range(winners.shape[0])] + [args.cond_negative_text for _ in range(losers.shape[0])]
                                # Re-tokenize for updated conditions
                                batch["cond_input_ids"] = tokenizer(cond_texts_updated, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                        elif args.jeremy_conditioning:
                            feed_pixel_values = torch.cat([winners, losers], dim=0)
                            cond_texts_updated = [args.cond_positive_text for _ in range(winners.shape[0])] + [args.cond_negative_text for _ in range(losers.shape[0])]
                            null_cond_texts = ["" for _ in range(winners.shape[0])]
                            # Re-tokenize for updated conditions
                            batch["cond_input_ids"] = tokenizer(cond_texts_updated, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                        else:
                            # Conditional DPO: condition selects preferred side
                            if args.multi_dim:
                                # Randomly select win or lose conditions for each example
                                # cond_input_ids has shape [batch_size, 2, seq_len]
                                cond_is_win = torch.randint(0, 2, (winners.shape[0],), device=winners.device)
                                # Select appropriate condition for each example: 0 for win, 1 for lose
                                selected_conds = []
                                for i in range(winners.shape[0]):
                                    selected_conds.append(batch["cond_input_ids"][i, cond_is_win[i], :])
                                batch["cond_input_ids"] = torch.stack(selected_conds)
                                # Don't concatenate here - let the repeat at line 1946 handle it
                            else:
                                cond_is_win = batch["cond_is_positive"]
                            cond_is_win_expanded = cond_is_win.view(-1, 1, 1, 1).to(winners.device)
                            first = torch.where(cond_is_win_expanded == 1, winners, losers)
                            second = torch.where(cond_is_win_expanded == 1, losers, winners)
                            feed_pixel_values = torch.cat([first, second], dim=0)
                            if (step == 0) and (global_step == 0) and accelerator.is_main_process:
                                debug_entries = []
                                for idx in range(min(8, winners.shape[0])):
                                    cond_flag = int(cond_is_win[idx].item())
                                    cond_label = args.cond_positive_text if cond_flag == 1 else args.cond_negative_text
                                    first_matches_winner = bool(torch.allclose(first[idx], winners[idx]))
                                    second_matches_loser = bool(torch.allclose(second[idx], losers[idx]))
                                    debug_entries.append({
                                        "cond_text": cond_label,
                                        "cond_is_win": cond_flag,
                                        "first_is_original_winner": first_matches_winner,
                                        "second_is_original_loser": second_matches_loser,
                                    })
                                accelerator.print(f"[CDPO DEBUG] Batch order check: {debug_entries}")
                elif args.train_method in ['sft', 'csft']:
                    feed_pixel_values = batch["pixel_values"]
                
                #### Diffusion Stuff ####
                # encode pixels --> latents
                with torch.no_grad():
                    latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # variants of noising
                if args.noise_offset: # haven't tried yet
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation: # haven't tried yet
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                    
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # only first 20% timesteps for SDXL refiner
                if 'refiner' in args.pretrained_model_name_or_path:
                    timesteps = timesteps % 200
                elif 'turbo' in args.pretrained_model_name_or_path:
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249
                
                if args.train_method in ['dpo', 'cdpo']: # make timesteps and noise same for pairs in DPO/CDPO
                    timesteps = timesteps.chunk(2)[0].repeat(2)
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)
                    if args.simultaneous_conditioning:
                        timesteps = timesteps.chunk(4)[0].repeat(4)
                        noise = noise.chunk(4)[0].repeat(4, 1, 1, 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                
                target_win = target_lose = None
                jeremy_win_timesteps = jeremy_lose_timesteps = None

                noisy_latents = noise_scheduler.add_noise(latents,
                                                          new_noise if args.input_perturbation else noise,
                                                          timesteps)
                if args.jeremy_conditioning:
                    jeremy_win_timesteps, jeremy_lose_timesteps = timesteps.chunk(2)
                    target_win, target_lose = noise.chunk(2)
                ### START PREP BATCH ###
                if args.sdxl:
                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        # Need to compute "time_ids" https://github.com/huggingface/diffusers/blob/v0.20.0-release/examples/text_to_image/train_text_to_image_sdxl.py#L969
                        # for SDXL-base these are torch.tensor([args.resolution, args.resolution, *crop_coords_top_left, *target_size))
                        if 'refiner' in args.pretrained_model_name_or_path:
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          6.0], # aesthetics conditioning https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L691C9-L691C24
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        else: # SDXL-base
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          args.resolution, 
                                                         args.resolution],
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        prompt_batch = encode_prompt_sdxl(batch, 
                                                          text_encoders,
                                                           tokenizers,
                                                           args.proportion_empty_prompts, 
                                                          caption_column='caption',
                                                           is_train=True,
                                                          )
                    if args.train_method in ['dpo', 'cdpo']:
                        prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
                        prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
                    unet_added_conditions = {"time_ids": add_time_ids,
                                            "text_embeds": prompt_batch["pooled_prompt_embeds"]}
                else: # sd1.5
                    # Get the text embedding for conditioning
                    if args.train_method in ['csft']:
                        # Already flattened to 2*B in collate
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    else:
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        if args.train_method in ['dpo', 'cdpo']:
                            if not args.jeremy_conditioning:
                                encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                    # Append condition tokens if conditional methods
                    if args.train_method in ['csft', 'cdpo']:
                        if args.train_method == 'cdpo':
                            cond_input_ids = batch["cond_input_ids"]
                            # Build once for B, then repeat to 2B to align with encoder_hidden_states
                            # cond_tokens = build_condition_tokens(cond_adapter, tokenizer, text_encoder, cond_texts, accelerator.device, encoder_hidden_states.dtype)
                            # cond_tokens = cond_tokens.repeat(2, 1, 1)
                            cond_tokens = text_encoder(cond_input_ids.to(accelerator.device))[0].to(encoder_hidden_states.dtype)
                            if not args.simultaneous_conditioning and not args.jeremy_conditioning:
                                cond_tokens = cond_tokens.repeat(2, 1, 1)
                            if args.jeremy_conditioning:
                                null_cond_tokens = text_encoder(tokenizer(null_cond_texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device))[0].to(encoder_hidden_states.dtype)
                        else:  # csft (already 2*B)
                            cond_input_ids = batch["cond_input_ids"]
                            # cond_tokens = build_condition_tokens(cond_adapter, tokenizer, text_encoder, cond_texts, accelerator.device, encoder_hidden_states.dtype)
                            cond_tokens = text_encoder(cond_input_ids.to(accelerator.device))[0].to(encoder_hidden_states.dtype)
                            # cond_tokens = cond_adapter(cond_tokens)
                        # encoder_hidden_states = torch.cat([encoder_hidden_states, cond_tokens], dim=1)
                #### END PREP BATCH ####
                        
                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise
                if args.simultaneous_conditioning:
                    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)

                if args.ip_adapter:
                    if args.jeremy_conditioning:
                        # model_pred = unet(
                        #     noisy_latents,
                        #     timesteps,
                        #     encoder_hidden_states, # TODO: only support SD1.5 for now
                        #     cond_tokens
                        # )
                        win_noisy_latent, lose_noisy_latent = noisy_latents.chunk(2)
                        win_cond_tokens, lose_cond_tokens = cond_tokens.chunk(2)
                        win_model_pred = unet(
                            win_noisy_latent,
                            jeremy_win_timesteps,
                            encoder_hidden_states,
                            win_cond_tokens
                        )
                        win_model_pred_no_cond = unet(
                            win_noisy_latent,
                            jeremy_win_timesteps,
                            encoder_hidden_states,
                            null_cond_tokens
                        )
                        lose_model_pred = unet(
                            lose_noisy_latent,
                            jeremy_lose_timesteps,
                            encoder_hidden_states,
                            lose_cond_tokens
                        )
                        lose_model_pred_no_cond = unet(
                            lose_noisy_latent,
                            jeremy_lose_timesteps,
                            encoder_hidden_states,
                            null_cond_tokens
                        )
                    else:
                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states, # TODO: only support SD1.5 for now
                            cond_tokens
                        )
                else:               
                # Make the prediction from the model we're learning
                    model_batch_args = (noisy_latents,
                                        timesteps, 
                                        prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states)
                    added_cond_kwargs = unet_added_conditions if args.sdxl else None
                
                    if cond_adapter is not None:
                        adapter_module = accelerator.unwrap_model(cond_adapter)
                        adapter_dtype = next(adapter_module.parameters()).dtype
                        cond_tokens = cond_tokens.to(adapter_dtype)
                        cond_tokens = cond_adapter(cond_tokens)
                        cond_tokens = cond_tokens.to(encoder_hidden_states.dtype)
                        encoder_hidden_states = torch.cat([encoder_hidden_states, cond_tokens], dim=1)
                    model_pred = unet(
                                    *model_batch_args,
                                    added_cond_kwargs = added_cond_kwargs
                                    ).sample
                #### START LOSS COMPUTATION ####
                if args.train_method in ['sft', 'csft']: # SFT/CSFT, casting for F.mse_loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.train_method in ['dpo']:
                    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                    # losses are both 2 * LBS
                    # 1st half of tensors is preferred (y_w), second half is unpreferred
                    model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    # below for logging purposes
                    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                    
                    model_diff = model_losses_w - model_losses_l # These are both LBS (as is t)
                    
                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        if args.ip_adapter:
                            ref_pred = ref_unet(
                                        noisy_latents,
                                        timesteps,
                                        encoder_hidden_states, # TODO: only support SD1.5 for now
                                    ).sample.detach()
                        else:
                            ref_pred = ref_unet(
                                        *model_batch_args,
                                          added_cond_kwargs = added_cond_kwargs
                                         ).sample.detach()
                        ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l
                        raw_ref_loss = ref_losses.mean()    
                        
                    scale_term = -0.5 * args.beta_dpo
                    inside_term = scale_term * (model_diff - ref_diff)
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    loss = -1 * F.logsigmoid(inside_term).mean()
                elif args.train_method in ['cdpo']:
                    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                    # losses are both 2 * LBS
                    # 1st half of tensors is preferred (y_w), second half is unpreferred
                    if args.jeremy_conditioning:
                        win_cond_losses = (win_model_pred - target_win).pow(2).mean(dim=[1,2,3])
                        lose_cond_losses = (lose_model_pred - target_lose).pow(2).mean(dim=[1,2,3])
                        win_no_cond_losses = (win_model_pred_no_cond - target_win).pow(2).mean(dim=[1,2,3])
                        lose_no_cond_losses = (lose_model_pred_no_cond - target_lose).pow(2).mean(dim=[1,2,3])

                        raw_model_loss = 0.25 * (win_cond_losses.mean() + lose_cond_losses.mean() + win_no_cond_losses.mean() + lose_no_cond_losses.mean())

                        model_diff_w = win_cond_losses - win_no_cond_losses
                        model_diff_l = lose_cond_losses - lose_no_cond_losses
                        model_diff = torch.cat([model_diff_w, model_diff_l], dim=0)

                    else:
                        model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                        model_losses_w, model_losses_l = model_losses.chunk(2)
                        # below for logging purposes
                        raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                        
                        model_diff = model_losses_w - model_losses_l # These are both LBS (as is t)
                    
                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        if args.ip_adapter:
                            if args.jeremy_conditioning:
                                win_ref_pred = ref_unet(
                                    win_noisy_latent,
                                    jeremy_win_timesteps,
                                    encoder_hidden_states, # TODO: only support SD1.5 for now
                                ).sample.detach()
                                lose_ref_pred = ref_unet(
                                    lose_noisy_latent,
                                    jeremy_lose_timesteps,
                                    encoder_hidden_states,
                                ).sample.detach()
                            else:
                                if ref_unet.__class__.__name__ == "IPAdapter":
                                    ref_pred = ref_unet(
                                        noisy_latents,
                                        timesteps,
                                        encoder_hidden_states,
                                        cond_tokens
                                    ).detach()
                                else:
                                    ref_pred = ref_unet(
                                                noisy_latents,
                                                timesteps,
                                                encoder_hidden_states, # TODO: only support SD1.5 for now
                                            ).sample.detach()
                        else:
                            ref_pred = ref_unet(
                                        *model_batch_args,
                                          added_cond_kwargs = added_cond_kwargs
                                         ).sample.detach()
                        if args.jeremy_conditioning:
                            ref_losses_w = (win_ref_pred - target_win).pow(2).mean(dim=[1,2,3])
                            ref_losses_l = (lose_ref_pred - target_lose).pow(2).mean(dim=[1,2,3])
                            ref_diff_w = ref_losses_w - ref_losses_w
                            ref_diff_l = ref_losses_l - ref_losses_l
                            ref_diff = torch.cat([ref_diff_w, ref_diff_l], dim=0)
                            raw_ref_loss = 0.5 * (ref_losses_w.mean() + ref_losses_l.mean())
                        else:
                            ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                            ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                            ref_diff = ref_losses_w - ref_losses_l
                            raw_ref_loss = ref_losses.mean()    
                    
                    if args.jeremy_conditioning:
                        scale_term = -0.5 * args.beta_dpo
                        win_diff = (win_cond_losses - ref_losses_w).pow(2) - (win_no_cond_losses - ref_losses_w).pow(2)
                        lose_diff = (lose_cond_losses - ref_losses_l).pow(2) - (lose_no_cond_losses - ref_losses_l).pow(2)
                        model_diff = torch.cat([win_diff, lose_diff], dim=0)
                        inside_term = scale_term * model_diff
                        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                        loss = -1 * F.logsigmoid(inside_term).mean()
                    else:
                        scale_term = -0.5 * args.beta_dpo
                        inside_term = scale_term * (model_diff - ref_diff)
                        # if args.simultaneous_conditioning:
                        #     bs = inside_term.shape[0]//2
                        #     inside_term[bs:] *= 0.1
                        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                        loss = -1 * F.logsigmoid(inside_term).mean()
                #### END LOSS COMPUTATION ###
                    
                # Gather the losses across all processes for logging 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method in ['dpo', 'cdpo']:
                    avg_model_lose_mse = accelerator.gather(model_losses_l.repeat(args.train_batch_size)).mean().item()
                    avg_model_win_mse = accelerator.gather(model_losses_w.repeat(args.train_batch_size)).mean().item()
                    avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                    avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        # Clip only trainable params; if csft_cond_only, UNet may be frozen
                        trainable_params = [p for p in unet.parameters() if p.requires_grad]
                        if len(trainable_params) > 0:
                            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.train_method in ['dpo', 'cdpo']:
                    accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                    accelerator.log({"ref_mse_unaccumulated": avg_ref_mse}, step=global_step)
                    accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                    accelerator.log({"model_lose_mse_accumulated": avg_model_lose_mse}, step=global_step)
                    accelerator.log({"model_win_mse_accumulated": avg_model_win_mse}, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                # Qualitative sampling every N steps (main process only)
                if accelerator.is_main_process and (global_step % SAMPLE_EVERY_STEPS == 0):
                    _build_or_update_sample_pipeline()
                    # Log a quick weight checksum to verify model weight changes
                    try:
                        _unet_chk = next(accelerator.unwrap_model(unet).parameters()).detach().float()
                        accelerator.log({"unet_first_weight_mean": _unet_chk.mean().item()}, step=global_step)

                        if args.train_method in ['csft', 'cdpo']:
                            cond_adapter_chk = next(accelerator.unwrap_model(cond_adapter).parameters()).detach().float()
                            accelerator.log({"cond_adapter_first_weight_mean": cond_adapter_chk.mean().item()}, step=global_step)
                    except Exception:
                        pass
                    with torch.inference_mode():
                        _gen = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        if args.sdxl:
                            images = sample_pipe(
                                prompt=SAMPLE_PROMPTS,
                                num_inference_steps=SAMPLE_INFERENCE_STEPS,
                                guidance_scale=SAMPLE_GUIDANCE_SCALE,
                                generator=_gen,
                            ).images
                        else:
                            if args.ip_adapter:
                                images = sample_pipe.__call__(
                                    self=sample_pipe,
                                    prompt=SAMPLE_PROMPTS,
                                    num_inference_steps=SAMPLE_INFERENCE_STEPS,
                                    guidance_scale=SAMPLE_GUIDANCE_SCALE,
                                    generator=_gen,
                                    positive_condition=args.cond_positive_text,
                                    negative_condition=args.cond_negative_text,
                                ).images
                                images2 = sample_pipe.__call__(
                                    self=sample_pipe,
                                    prompt=SAMPLE_PROMPTS,
                                    num_inference_steps=SAMPLE_INFERENCE_STEPS,
                                    guidance_scale=SAMPLE_GUIDANCE_SCALE2,
                                    generator=_gen,
                                    positive_condition=args.cond_positive_text,
                                    negative_condition=args.cond_negative_text,
                                ).images
                                images3 = sample_pipe.__call__(
                                    self=sample_pipe,
                                    prompt=SAMPLE_PROMPTS,
                                    num_inference_steps=SAMPLE_INFERENCE_STEPS,
                                    guidance_scale=1.0,
                                    generator=_gen,
                                    positive_condition=args.cond_positive_text,
                                    negative_condition=args.cond_negative_text,
                                ).images
                            else:
                                images = sample_pipe(
                                    prompt=SAMPLE_PROMPTS,
                                    num_inference_steps=SAMPLE_INFERENCE_STEPS,
                                    guidance_scale=SAMPLE_GUIDANCE_SCALE,
                                    generator=_gen,
                                ).images
                                images2 = sample_pipe(
                                    prompt=SAMPLE_PROMPTS,
                                    num_inference_steps=SAMPLE_INFERENCE_STEPS,
                                    guidance_scale=SAMPLE_GUIDANCE_SCALE2,
                                    generator=_gen,
                                ).images
                    # Log list of images to trackers (e.g., W&B)
                    try:
                        if is_wandb_available():
                            wandb_images = [wandb.Image(img) for img in images]
                            accelerator.log({"samples": wandb_images}, step=global_step)
                            wandb_images2 = [wandb.Image(img) for img in images2]
                            accelerator.log({"samples2": wandb_images2}, step=global_step)
                            wandb_images3 = [wandb.Image(img) for img in images3]
                            accelerator.log({"cfg_1": wandb_images3}, step=global_step)
                        else:
                            # Fallback: save to disk under output_dir/samples
                            out_dir = os.path.join(args.output_dir, "samples")
                            os.makedirs(out_dir, exist_ok=True)
                            for idx, img in enumerate(images):
                                img.save(os.path.join(out_dir, f"step{global_step:06d}_{idx}.png"))
                    except Exception as e:
                        logger.warning(f"Failed to log samples at step {global_step}: {e}")

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info("Pretty sure saving/loading is fixed but proceed cautiously")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method in ['dpo', 'cdpo']:
                logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = accelerator.unwrap_model(unet)
    #     if args.sdxl:
    #         # Serialize pipeline.
    #         vae = AutoencoderKL.from_pretrained(
    #             vae_path,
    #             subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
    #             revision=args.revision,
    #             torch_dtype=weight_dtype,
    #         )
    #         pipeline = StableDiffusionXLPipeline.from_pretrained(
    #             args.pretrained_model_name_or_path, unet=unet, vae=vae, revision=args.revision, torch_dtype=weight_dtype
    #         )
    #         pipeline.save_pretrained(args.output_dir)
    #     else:
    #         pipeline = StableDiffusionPipeline.from_pretrained(
    #             args.pretrained_model_name_or_path,
    #             text_encoder=text_encoder,
    #             vae=vae,
    #             unet=unet,
    #             revision=args.revision,
    #         )
    #     pipeline.save_pretrained(args.output_dir)
    #     # Save conditional adapter alongside pipeline for SD1.5 conditional methods
    #     if (not args.sdxl) and (cond_adapter is not None) and (args.train_method in ["csft", "cdpo"]) and (not args.ip_adapter):
    #         try:
    #             ca = accelerator.unwrap_model(cond_adapter)
    #         except Exception:
    #             ca = cond_adapter
    #         ca.save_pretrained(os.path.join(args.output_dir, "cond_adapter"))


    accelerator.end_training()


if __name__ == "__main__":
    main()
