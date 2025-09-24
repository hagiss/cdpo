import os
import json
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from einops import rearrange, repeat

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class SD15ConditionAdapter(nn.Module):
    """
    Projects a pooled CLIP text embedding for a condition (e.g. "win"/"lose")
    into one or more pseudo-token embeddings to be appended to the prompt
    encoder hidden states for SD1.5 UNet cross-attention.

    The final projection layer is zero-initialized so the adapter is a no-op at init.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        projector_type: str = "linear",
        mlp_hidden_dim: Optional[int] = None,
        num_condition_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.projector_type = projector_type
        self.mlp_hidden_dim = mlp_hidden_dim or hidden_size
        self.num_condition_tokens = num_condition_tokens

        if projector_type == "linear":
            self.projector = nn.Linear(hidden_size, hidden_size * num_condition_tokens, bias=True)
            nn.init.zeros_(self.projector.weight)
            nn.init.zeros_(self.projector.bias)
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(hidden_size, self.mlp_hidden_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.mlp_hidden_dim, hidden_size, bias=True),
            )
            # zero-init the final layer only, so it's a no-op initially
            final: nn.Linear = self.projector[-1]  # type: ignore[index]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")

    def forward(self, embed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embed_tokens: Tensor of shape [batch, seq_len, hidden_size]

        Returns:
            condition_token_embeds: Tensor of shape [batch, seq_len, hidden_size]
        """
        batch = embed_tokens.shape[0]
        embed_tokens = rearrange(embed_tokens, "b s d -> (b s) d")
        proj = self.projector(embed_tokens)  # [(b s), hidden_size]
        proj = rearrange(proj, "(b s) d -> b s d", b=batch)
        return proj

    # Simple serialization helpers to store adapter independent of diffusers
    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        config = {
            "hidden_size": self.hidden_size,
            "projector_type": self.projector_type,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "num_condition_tokens": self.num_condition_tokens,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "SD15ConditionAdapter":
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model


@torch.no_grad()
def encode_condition_pooled(
    tokenizer,
    text_encoder,
    condition_texts: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Tokenize and encode condition texts using the same tokenizer/text_encoder as prompts.

    Returns a pooled representation [batch, hidden_size] by taking the first token
    embedding (CLIP-like) from the last hidden state.
    """
    text_inputs = tokenizer(
        condition_texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    outputs = text_encoder(input_ids)
    last_hidden_state = outputs[0].to(dtype)
    pooled = last_hidden_state[:, 0, :]  # [B, hidden]
    return pooled


# def build_condition_tokens(
#     adapter: SD15ConditionAdapter,
#     tokenizer,
#     text_encoder,
#     condition_texts: List[str],
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> torch.Tensor:
#     """
#     Produces condition token embeddings [batch, seq_len, hidden_size] to be
#     concatenated with prompt encoder hidden states.
#     """
#     # Compute in adapter parameter dtype to avoid matmul dtype conflicts, then cast to requested dtype
#     tokens = adapter(embed_tokens)
    
#     # Ensure dtype/device match caller expectations
#     return tokens.to(device=device, dtype=dtype)


# def build_cfg_condition_embeddings(
#     adapter: SD15ConditionAdapter,
#     tokenizer,
#     text_encoder,
#     positive_condition: str,
#     negative_condition: str,
#     batch_size: int,
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Utility for inference: returns (pos_cond_tokens, neg_cond_tokens) each of shape
#     [batch, num_tokens, hidden_size].
#     """
#     pos_list = [positive_condition] * batch_size
#     neg_list = [negative_condition] * batch_size
#     pos = build_condition_tokens(adapter, tokenizer, text_encoder, pos_list, device, dtype)
#     neg = build_condition_tokens(adapter, tokenizer, text_encoder, neg_list, device, dtype)
#     return pos, neg


def monkey_patch_sd15_pipeline_for_condition(
    pipe,
    adapter: SD15ConditionAdapter,
    positive_condition: str = "win",
    negative_condition: str = "lose",
):
    """
    Monkey-patches a StableDiffusionPipeline (SD1.5) to append condition tokens
    to both conditional and unconditional (CFG) prompt embeddings.
    """
    if not hasattr(pipe, "tokenizer") or not hasattr(pipe, "text_encoder"):
        raise ValueError("Pipeline missing tokenizer/text_encoder attributes required for conditioning")

    pipe.condition_adapter = adapter
    pipe.cond_positive_text = positive_condition
    pipe.cond_negative_text = negative_condition

    # Preferred patch path: wrap encode_prompt when available
    # if hasattr(pipe, "encode_prompt"):
    #     if getattr(pipe, "_orig_encode_prompt", None) is None:
    #         pipe._orig_encode_prompt = pipe.encode_prompt

    #     def encode_prompt_with_condition(*args, **kwargs):
    #         outputs = pipe._orig_encode_prompt(*args, **kwargs)
    #         # Diffusers returns either prompt_embeds alone, or (prompt_embeds, negative_prompt_embeds)
    #         if isinstance(outputs, tuple):
    #             prompt_embeds, negative_prompt_embeds = outputs[:2]
    #             extra = outputs[2:]
    #         else:
    #             prompt_embeds, negative_prompt_embeds = outputs, None
    #             extra = tuple()

    #         device = prompt_embeds.device
    #         dtype = prompt_embeds.dtype
    #         batch = prompt_embeds.shape[0]
    #         # Build condition tokens for both branches
    #         pos_tokens = build_condition_tokens(pipe.condition_adapter, pipe.tokenizer, pipe.text_encoder,
    #                                             [pipe.cond_positive_text] * batch, device, dtype)
    #         prompt_embeds = torch.cat([prompt_embeds, pos_tokens], dim=1)
    #         # Debug confirmation for encode_prompt patch path
    #         try:
    #             print("[COND-PATCH] encode_prompt: prompt_embeds+pos_tokens ->",
    #                   tuple(prompt_embeds.shape), tuple(pos_tokens.shape))
    #         except Exception:
    #             pass
    #         neg_tokens = build_condition_tokens(pipe.condition_adapter, pipe.tokenizer, pipe.text_encoder,
    #                                                 [pipe.cond_negative_text] * batch, device, prompt_embeds.dtype)
    #         if negative_prompt_embeds is not None:
    #             # batch_u = negative_prompt_embeds.shape[0]
    #             # neg_tokens = build_condition_tokens(pipe.condition_adapter, pipe.tokenizer, pipe.text_encoder,
    #             #                                     [pipe.cond_negative_text] * batch_u, device, negative_prompt_embeds.dtype)
    #             negative_prompt_embeds = torch.cat([negative_prompt_embeds, neg_tokens], dim=1)
    #             try:
    #                 print("[COND-PATCH] encode_prompt: negative_embeds+neg_tokens ->",
    #                       tuple(negative_prompt_embeds.shape), tuple(neg_tokens.shape))
    #             except Exception:
    #                 pass
    #         else:
    #             negative_prompt_embeds = neg_tokens

    #         return (prompt_embeds, negative_prompt_embeds, *extra) if negative_prompt_embeds is not None else prompt_embeds

    #     pipe.encode_prompt = encode_prompt_with_condition

    # Some versions use _encode_prompt internally; patch it too if present
    if hasattr(pipe, "_encode_prompt"):
        if getattr(pipe, "_orig__encode_prompt", None) is None:
            pipe._orig__encode_prompt = pipe._encode_prompt

        def _encode_prompt_with_condition(*args, **kwargs):
            outputs = pipe._orig__encode_prompt(*args, **kwargs)

            # In SD1.5 pipelines, _encode_prompt returns a single Tensor. Preserve that.
            if isinstance(outputs, tuple):
                prompt_embeds = outputs[0]
            else:
                prompt_embeds = outputs

            device = prompt_embeds.device
            dtype = prompt_embeds.dtype

            # Try to obtain do_classifier_free_guidance from args/kwargs; fallback to shape heuristic
            do_cfg = None
            if "do_classifier_free_guidance" in kwargs:
                do_cfg = bool(kwargs["do_classifier_free_guidance"])
            elif len(args) >= 4:
                # args: prompt, device, num_images_per_prompt, do_classifier_free_guidance, ...
                do_cfg = bool(args[3])
            if do_cfg is None:
                do_cfg = (prompt_embeds.shape[0] % 2 == 0)

            if do_cfg and prompt_embeds.shape[0] % 2 == 0:
                # Replace prompt and negative_prompt in args/kwargs with cond_positive_text and cond_negative_text
                # Convert args to list to allow modification
                args = list(args)
                # Replace prompt (first positional argument)
                if len(args) > 0:
                    args[0] = pipe.cond_positive_text
                if "prompt" in kwargs:
                    kwargs["prompt"] = pipe.cond_positive_text
                # Replace negative_prompt (fifth positional argument)
                if len(args) > 4:
                    args[4] = pipe.cond_negative_text
                if "negative_prompt" in kwargs:
                    kwargs["negative_prompt"] = pipe.cond_negative_text
                cond_embeds = pipe._orig__encode_prompt(*args, **kwargs)
                cond_embeds = pipe.condition_adapter(cond_embeds)

                pos_cond_embeds = repeat(cond_embeds[0], 's d -> b s d', b=prompt_embeds.shape[0]//2)
                neg_cond_embeds = repeat(cond_embeds[1], 's d -> b s d', b=prompt_embeds.shape[0]//2)
                cond_embeds = torch.cat([pos_cond_embeds, neg_cond_embeds], dim=0)
                prompt_embeds = torch.cat([prompt_embeds, cond_embeds], dim=1)


                # Append negative condition to uncond half, positive to cond half
                # half = prompt_embeds.shape[0] // 2
                # uncond = prompt_embeds[:half]
                # cond = prompt_embeds[half:]

                # neg_tokens = build_condition_tokens(
                #     pipe.condition_adapter,
                #     pipe.tokenizer,
                #     pipe.text_encoder,
                #     [pipe.cond_negative_text] * half,
                #     device,
                #     dtype,
                # )
                # pos_tokens = build_condition_tokens(
                #     pipe.condition_adapter,
                #     pipe.tokenizer,
                #     pipe.text_encoder,
                #     [pipe.cond_positive_text] * half,
                #     device,
                #     dtype,
                # )

                # uncond = torch.cat([uncond, neg_tokens], dim=1)
                # cond = torch.cat([cond, pos_tokens], dim=1)
                # prompt_embeds = torch.cat([uncond, cond], dim=0)

                # try:
                #     print("[COND-PATCH] _encode_prompt (CFG):",
                #           "uncond+neg_tokens ->", tuple(uncond.shape), tuple(neg_tokens.shape), pipe.cond_negative_text,
                #           "| cond+pos_tokens ->", tuple(cond.shape), tuple(pos_tokens.shape), pipe.cond_positive_text)
                # except Exception:
                #     pass
            # else:
            #     # No CFG: only conditional branch exists; append positive tokens
            #     batch = prompt_embeds.shape[0]
            #     pos_tokens = build_condition_tokens(
            #         pipe.condition_adapter,
            #         pipe.tokenizer,
            #         pipe.text_encoder,
            #         [pipe.cond_positive_text] * batch,
            #         device,
            #         dtype,
            #     )
            #     prompt_embeds = torch.cat([prompt_embeds, pos_tokens], dim=1)
            #     try:
            #         print("[COND-PATCH] _encode_prompt (no-CFG): prompt+pos_tokens ->",
            #               tuple(prompt_embeds.shape), tuple(pos_tokens.shape), pipe.cond_positive_text)
            #     except Exception:
            #         pass

            # Always return a Tensor to match SD1.5 _encode_prompt contract
            return prompt_embeds

        pipe._encode_prompt = _encode_prompt_with_condition

    else:
        # Fallback: wrap __call__ to inject conditional embeddings when only text prompts are provided
        if getattr(pipe, "_orig_call", None) is None:
            pipe._orig_call = pipe.__call__

        def __call_with_condition__(*args, **kwargs):
            # If caller supplies prompt_embeds, extend them and optional negative with cond tokens
            prompt_embeds = kwargs.get("prompt_embeds", None)
            negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
            device = kwargs.get("device", next(pipe.unet.parameters()).device)
            dtype = next(pipe.unet.parameters()).dtype

            if prompt_embeds is None:
                # Build from text prompt
                prompts = kwargs.get("prompt", None)
                if prompts is None:
                    return pipe._orig_call(*args, **kwargs)
                if isinstance(prompts, str):
                    prompts = [prompts]
                tokenized = pipe.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(pipe.text_encoder.device)
                prompt_embeds = pipe.text_encoder(input_ids)[0].to(dtype=dtype)
                kwargs.pop("prompt", None)

            # Negative branch
            if negative_prompt_embeds is None:
                neg_prompts = kwargs.get("negative_prompt", "")
                if isinstance(neg_prompts, str):
                    neg_prompts = [neg_prompts] * prompt_embeds.shape[0]
                tokenized = pipe.tokenizer(
                    neg_prompts,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                neg_ids = tokenized.input_ids.to(pipe.text_encoder.device)
                negative_prompt_embeds = pipe.text_encoder(neg_ids)[0].to(dtype=dtype)
                kwargs.pop("negative_prompt", None)

            # Append condition tokens
            batch = prompt_embeds.shape[0]
            pos_tokens = build_condition_tokens(pipe.condition_adapter, pipe.tokenizer, pipe.text_encoder,
                                                [pipe.cond_positive_text] * batch, prompt_embeds.device, prompt_embeds.dtype)
            prompt_embeds = torch.cat([prompt_embeds, pos_tokens], dim=1)

            batch_u = negative_prompt_embeds.shape[0]
            neg_tokens = build_condition_tokens(pipe.condition_adapter, pipe.tokenizer, pipe.text_encoder,
                                                [pipe.cond_negative_text] * batch_u, negative_prompt_embeds.device, negative_prompt_embeds.dtype)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, neg_tokens], dim=1)
            try:
                print("[COND-PATCH] __call__: negative_embeds+neg_tokens ->",
                      tuple(negative_prompt_embeds.shape), tuple(neg_tokens.shape))
            except Exception:
                pass

            kwargs["prompt_embeds"] = prompt_embeds
            kwargs["negative_prompt_embeds"] = negative_prompt_embeds
            # Ensure we don't pass raw prompts anymore
            kwargs.pop("prompt", None)
            kwargs.pop("negative_prompt", None)
            return pipe._orig_call(*args, **kwargs)

        pipe.__call__ = __call_with_condition__
    return pipe




def monkey_patch_sd15_pipeline_for_condition_CA(
    pipe,
    adapter: SD15ConditionAdapter,
):
    """
    Monkey-patches a StableDiffusionPipeline (SD1.5) to append condition tokens
    to both conditional and unconditional (CFG) prompt embeddings.
    """
    if not hasattr(pipe, "tokenizer") or not hasattr(pipe, "text_encoder"):
        raise ValueError("Pipeline missing tokenizer/text_encoder attributes required for conditioning")

    pipe.condition_adapter = adapter

    @torch.no_grad()
    def __call_with_condition_CA__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        positive_condition: Optional[str] = None,
        negative_condition: Optional[str] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        if positive_condition is not None:
            cond_embeds = self._encode_prompt(
                positive_condition,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_condition,
                lora_scale=text_encoder_lora_scale,
            )
            cond_embeds = self.condition_adapter(cond_embeds)
            cross_attention_kwargs.update({"condition_embeds": cond_embeds})

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    pipe.__call__ = __call_with_condition_CA__
    return pipe
