import os
import json
import math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineOutput, StableDiffusionXLPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from einops import rearrange, repeat
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
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


class SDXLConditionAdapter(nn.Module):
    """
    Projects concatenated CLIP text embeddings for a condition (e.g. "win"/"lose")
    into one or more pseudo-token embeddings to be appended to the prompt
    encoder hidden states for SDXL UNet cross-attention.
    
    SDXL uses two text encoders: text_encoder_1 (768 dim) + text_encoder_2 (1280 dim).
    Input embeddings are concatenated: 768 + 1280 = 2048 dim.

    The final projection layer is zero-initialized so the adapter is a no-op at init.
    """

    def __init__(
        self,
        hidden_size: int = 2048,  # Default: 768 (text_encoder_1) + 1280 (text_encoder_2)
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
            self.projector = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.projector.weight)
            nn.init.zeros_(self.projector.bias)
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(hidden_size, self.mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.mlp_hidden_dim, hidden_size),
            )
            # zero-init the final layer only, so it's a no-op initially
            final: nn.Linear = self.projector[-1]  # type: ignore[index]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")
        self.norm = torch.nn.LayerNorm(hidden_size)

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
        proj = self.norm(proj)
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
    def from_pretrained(cls, load_directory: str, projector_type: str = "linear", mlp_hidden_dim: int = 4096) -> "SDXLConditionAdapter":
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(**config)
        model.projector_type = projector_type
        model.mlp_hidden_dim = mlp_hidden_dim
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model


class IPAdapter_SDXL(torch.nn.Module):
    """IP-Adapter for SDXL with condition token support.
    
    Note: For CSFT/CDPO, the 'condition_adapter' is the SDXLConditionAdapter that projects
    condition embeddings (e.g., "win"/"lose") into tokens to be concatenated with prompt_embeds.
    """
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        # For CSFT/CDPO: condition_adapter is SDXLConditionAdapter
        self.image_proj_model = image_proj_model.to(device=unet.device, dtype=unet.dtype)
        self.adapter_modules = adapter_modules.to(device=unet.device, dtype=unet.dtype)

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    @property
    def dtype(self):
        # Expose a dtype property expected by diffusers Pipeline.to
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return getattr(self.unet, "dtype", torch.float32)

    @property
    def device(self):
        # Expose a device property for convenience/compatibility
        try:
            return next(self.parameters()).device
        except StopIteration:
            return getattr(self.unet, "device", torch.device("cpu"))

    @property
    def config(self):
        return self.unet.config

    def forward(self, noisy_latents, timesteps, prompt_embeds, cond_embeds=None, added_cond_kwargs=None):
        """
        Forward pass for SDXL UNet with optional condition embeddings.
        
        Args:
            noisy_latents: Noisy latent tensors [B, C, H, W]
            timesteps: Timestep tensor [B]
            prompt_embeds: Combined text embeddings from both SDXL encoders [B, seq_len, dim]
                          (concatenation of text_encoder_1 and text_encoder_2 outputs)
            cond_embeds: Optional condition embeddings [B, seq_len, hidden_size] 
                        These will be projected by condition_adapter and concatenated to prompt_embeds
            added_cond_kwargs: SDXL-specific kwargs dict with 'time_ids' and 'text_embeds' (pooled)
        
        Returns:
            noise_pred: Predicted noise [B, C, H, W]
        """
        # If condition embeddings are provided, project them and concatenate
        if cond_embeds is not None:
            # Project condition embeddings through the adapter
            cond_tokens = self.image_proj_model(cond_embeds.to(dtype=self.dtype))
            # Concatenate condition tokens to prompt embeddings
            prompt_embeds = torch.cat([prompt_embeds, cond_tokens], dim=1)
        
        # Call SDXL UNet with added_cond_kwargs for time_ids and pooled text_embeds
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            prompt_embeds,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        # def unet_forward(latent_input, timestep, prompt_embeds, added_cond_kwargs):
        #     return self.unet(
        #         latent_input, 
        #         timestep, 
        #         prompt_embeds,
        #         added_cond_kwargs=added_cond_kwargs
        #     ).sample

        # noise_pred = torch.utils.checkpoint.checkpoint(
        #     unet_forward,
        #     noisy_latents,
        #     timesteps,
        #     prompt_embeds,
        #     added_cond_kwargs,
        #     use_reentrant=True,
        # )
        
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_cond_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        ckpt_path = os.path.join(ckpt_path, "ip_adapter")
        cond_adapter_path = os.path.join(ckpt_path, "proj_model.pt")
        adapter_path = os.path.join(ckpt_path, "adapter_modules.pt")
        unet_path = os.path.join(ckpt_path, "unet.pt")
        unet_config_path = os.path.join(ckpt_path, "unet_config.json")
        if os.path.exists(unet_config_path):
            with open(unet_config_path, "r") as f:
                unet_config_dict = json.load(f)
            # Safely update UNet config without assuming a specific config class API
            if hasattr(self.unet, "register_to_config"):
                self.unet.register_to_config(**unet_config_dict)
            else:
                try:
                    # If config is a mutable mapping
                    self.unet.config.update(unet_config_dict)
                except Exception:
                    # As a last resort, reassign plain dict
                    self.unet.config = unet_config_dict

        state_dict_cond = torch.load(cond_adapter_path, map_location="cpu")
        state_dict_adapter = torch.load(adapter_path, map_location="cpu")

        if os.path.exists(unet_path):
            unet_state_dict = torch.load(unet_path, map_location="cpu")
            # Load UNet weights but allow missing IP-Adapter processor params,
            # which are stored separately in adapter_modules.pt
            self.unet.load_state_dict(unet_state_dict, strict=False)

        # Load state dict for condition_adapter and adapter_modules
        self.image_proj_model.load_state_dict(state_dict_cond, strict=True)
        self.adapter_modules.load_state_dict(state_dict_adapter, strict=True)

        # Calculate new checksums
        new_cond_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_cond_adapter_sum != new_cond_adapter_sum, "Weights of condition_adapter did not change!"
        # assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    def save_pretrained(self, save_directory: str, cond_only=True) -> None:
        os.makedirs(save_directory, exist_ok=True)
        torch.save(
            self.image_proj_model.state_dict(),
            os.path.join(save_directory, "proj_model.pt"),
        )
        torch.save(
            self.adapter_modules.state_dict(),
            os.path.join(save_directory, "adapter_modules.pt"),
        )

        if not cond_only:
            torch.save(
                self.unet.state_dict(),
                os.path.join(save_directory, "unet.pt"),
            )
            if hasattr(self.unet, "config"):
                with open(os.path.join(save_directory, "unet_config.json"), "w") as f:
                    cfg = self.unet.config
                    # Convert configs without to_dict (e.g., FrozenDict) into a serializable dict
                    def _to_serializable(obj):
                        if hasattr(obj, "to_dict"):
                            return _to_serializable(obj.to_dict())
                        # Mapping-like objects
                        if isinstance(obj, dict):
                            return {k: _to_serializable(v) for k, v in obj.items()}
                        if hasattr(obj, "items"):
                            try:
                                return {k: _to_serializable(v) for k, v in obj.items()}
                            except Exception:
                                pass
                        # Sequences
                        if isinstance(obj, (list, tuple)):
                            return [_to_serializable(v) for v in obj]
                        # Fallback: try to cast FrozenDict or similar to dict
                        try:
                            return dict(obj)
                        except Exception:
                            return obj
                    json.dump(_to_serializable(cfg), f)


def init_adapter_SDXL(unet):
    attn_procs = {}
    unet_sd = unet.state_dict()
    flag = False
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=77)
            attn_procs[name].load_state_dict(weights)
            attn_procs[name].add_cross_attention_to_latent()
            # if flag:
            #     attn_procs[name].add_cross_attention_to_latent()
            #     flag = False
            # else:
            #     flag = True
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules


def monkey_patch_sdxl_pipeline_for_ipadapter(pipe):

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    StableDiffusionXLPipeline._get_add_time_ids = _get_add_time_ids

    @torch.no_grad()
    def __call_with_ipadapter__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        positive_condition: Optional[str] = None,
        negative_condition: Optional[str] = None,
        decomposed_additive_guidance: Optional[bool] = False,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
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
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
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

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # Encode condition tokens if provided (for CSFT/CDPO sampling)
        cond_embeds = None
        if positive_condition is not None and negative_condition is not None:
            # Check if we have a image_proj_model (should be inside IPAdapter_SDXL wrapper)
            if hasattr(self.unet, 'image_proj_model'):
                # Encode condition texts using BOTH text encoders (matching training and SDXL architecture)
                # We concatenate text_encoder_1 (768) + text_encoder_2 (1280) = 2048 dim
                tokenizers = [self.tokenizer, self.tokenizer_2] if hasattr(self, 'tokenizer_2') else [self.tokenizer]
                text_encoders = [self.text_encoder, self.text_encoder_2] if hasattr(self, 'text_encoder_2') else [self.text_encoder]
                
                # Prepare condition texts: negative for uncond, positive for cond
                if do_classifier_free_guidance:
                    cond_texts = [negative_condition] * batch_size * num_images_per_prompt + \
                                 [positive_condition] * batch_size * num_images_per_prompt
                else:
                    cond_texts = [positive_condition] * batch_size * num_images_per_prompt
                
                cond_embeds_list = []
                
                with torch.no_grad():
                    # Encode with both text encoders, following SDXL's encode_prompt pattern
                    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                        text_inputs = tokenizer(
                            cond_texts,
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_input_ids = text_inputs.input_ids.to(device)
                        
                        outputs = text_encoder(
                            text_input_ids,
                            output_hidden_states=True,
                        )
                        
                        # Use hidden states (not pooled) to maintain sequence dimension
                        cond_embeds = outputs.hidden_states[-2]
                        cond_embeds_list.append(cond_embeds)
                    
                    # Concatenate embeddings from both encoders along the last dimension
                    # Result: [batch, seq_len, 768 + 1280] = [batch, seq_len, 2048]
                    cond_embeds = torch.concat(cond_embeds_list, dim=-1)
                
                # cond_embeds is now [2*batch*num_images, seq_len, 2048] for CFG
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                
                # Check if unet is IPAdapter_SDXL wrapper
                if hasattr(self.unet, 'image_proj_model') and cond_embeds is not None:
                    # Call IPAdapter_SDXL with condition embeddings
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        prompt_embeds,
                        cond_embeds=cond_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                    )
                else:
                    # Standard SDXL UNet call (no condition tokens)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
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

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    pipe.__call__ = __call_with_ipadapter__
    return pipe