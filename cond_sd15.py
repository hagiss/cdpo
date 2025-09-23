import os
import json
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class SD15ConditionAdapter(nn.Module):
    """
    Projects a pooled CLIP text embedding for a condition (e.g. "win"/"lose")
    into one or more pseudo-token embeddings to be appended to the prompt
    encoder hidden states for SD1.5 UNet cross-attention.

    The final projection layer is zero-initialized so the adapter is a no-op at init.
    """

    def __init__(
        self,
        hidden_size: int = 768,
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
                nn.Linear(self.mlp_hidden_dim, hidden_size * num_condition_tokens, bias=True),
            )
            # zero-init the final layer only, so it's a no-op initially
            final: nn.Linear = self.projector[-1]  # type: ignore[index]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")

    def forward(self, pooled_condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_condition: Tensor of shape [batch, hidden_size]

        Returns:
            condition_token_embeds: Tensor of shape [batch, num_tokens, hidden_size]
        """
        proj = self.projector(pooled_condition)  # [B, hidden_size * num_tokens]
        proj = proj.view(proj.shape[0], self.num_condition_tokens, self.hidden_size)
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


def build_condition_tokens(
    adapter: SD15ConditionAdapter,
    tokenizer,
    text_encoder,
    condition_texts: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Produces condition token embeddings [batch, num_tokens, hidden_size] to be
    concatenated with prompt encoder hidden states.
    """
    # Compute in adapter parameter dtype to avoid matmul dtype conflicts, then cast to requested dtype
    adapter_dtype = next(adapter.parameters()).dtype
    with torch.no_grad():
        pooled = encode_condition_pooled(tokenizer, text_encoder, condition_texts, device, adapter_dtype)
    tokens = adapter(pooled)
    # Ensure dtype/device match caller expectations
    return tokens.to(device=device, dtype=dtype)


def build_cfg_condition_embeddings(
    adapter: SD15ConditionAdapter,
    tokenizer,
    text_encoder,
    positive_condition: str,
    negative_condition: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility for inference: returns (pos_cond_tokens, neg_cond_tokens) each of shape
    [batch, num_tokens, hidden_size].
    """
    pos_list = [positive_condition] * batch_size
    neg_list = [negative_condition] * batch_size
    pos = build_condition_tokens(adapter, tokenizer, text_encoder, pos_list, device, dtype)
    neg = build_condition_tokens(adapter, tokenizer, text_encoder, neg_list, device, dtype)
    return pos, neg


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
                # Append negative condition to uncond half, positive to cond half
                half = prompt_embeds.shape[0] // 2
                uncond = prompt_embeds[:half]
                cond = prompt_embeds[half:]

                neg_tokens = build_condition_tokens(
                    pipe.condition_adapter,
                    pipe.tokenizer,
                    pipe.text_encoder,
                    [pipe.cond_negative_text] * half,
                    device,
                    dtype,
                )
                pos_tokens = build_condition_tokens(
                    pipe.condition_adapter,
                    pipe.tokenizer,
                    pipe.text_encoder,
                    [pipe.cond_positive_text] * half,
                    device,
                    dtype,
                )

                uncond = torch.cat([uncond, neg_tokens], dim=1)
                cond = torch.cat([cond, pos_tokens], dim=1)
                prompt_embeds = torch.cat([uncond, cond], dim=0)
                try:
                    print("[COND-PATCH] _encode_prompt (CFG):",
                          "uncond+neg_tokens ->", tuple(uncond.shape), tuple(neg_tokens.shape), pipe.cond_negative_text,
                          "| cond+pos_tokens ->", tuple(cond.shape), tuple(pos_tokens.shape), pipe.cond_positive_text)
                except Exception:
                    pass
            else:
                # No CFG: only conditional branch exists; append positive tokens
                batch = prompt_embeds.shape[0]
                pos_tokens = build_condition_tokens(
                    pipe.condition_adapter,
                    pipe.tokenizer,
                    pipe.text_encoder,
                    [pipe.cond_positive_text] * batch,
                    device,
                    dtype,
                )
                prompt_embeds = torch.cat([prompt_embeds, pos_tokens], dim=1)
                try:
                    print("[COND-PATCH] _encode_prompt (no-CFG): prompt+pos_tokens ->",
                          tuple(prompt_embeds.shape), tuple(pos_tokens.shape), pipe.cond_positive_text)
                except Exception:
                    pass

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


