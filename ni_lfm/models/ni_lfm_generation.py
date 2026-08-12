# Copyright 2025 IBM Corp.
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
# limitations under the License.

import random
import warnings

from functools import partial
from typing import Any

import torch

from einops import rearrange
from torch import nn

from .decoder_embeddings import ImageTokenDecoderEmbedding
from .encoder_embeddings import ImageEncoderEmbedding, ImageTokenEncoderEmbedding
from .generate import (
    GenerationSampler,
    build_chained_generation_schedules,
    init_cond_target_modality,
    init_empty_target_modality,
    init_full_input_modality,
)
from .ni_lfm import MODEL_CONFIGS, NILunarFM
from .lfm_utils import LayerNorm, init_enc_dec_embeddings

from ni_lfm.utils.tokenizer import EOS_TOKEN, S1_TOKEN


class NILunarFMGeneration(nn.Module):
    """Modified NILunarFM model for a "Thinking in Modalities" approach.

    Args:
        input_modalities (list[str]): Conditioning modality names.
        output_modalities (list[str]): Target modality names, generated in the given order.
        modality_info (dict): Per-modality info dict (embedding classes, kwargs, dataset params).
        tokenizers (nn.ModuleDict): Pretrained per-modality VQ / detokenizer modules.
        cfg: Pretraining config, forwarded to `init_enc_dec_embeddings`.
        text_tokenizer: HuggingFace `Tokenizer` for sequence modalities. Optional.
        decoding_steps (list, int): Number of decoding steps for each TiM modality. Defaults to 1.
        temps (list, float): Sampling temperatures for each TiM modality. Defaults to 1.0.
        top_p (float): Top-p sampling threshold for TiM modalities. Ignored if set to 0.0. Defaults to 0.8.
        top_k (int): Top-k sampling threshold for TiM modalities. Ignored if set to 0. Defaults to 0.
        timesteps (int): Number of denoising steps used by the detokenizers. Defaults to 50.
        dim (int): Patch embedding dimension.
        encoder_depth (int): Depth of ViT / number of encoder blocks.
        decoder_depth (int): Number of decoder blocks.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        proj_bias (bool): If True, adds a bias to the attention out proj layer.
        mlp_bias (bool): If True, adds a learnable bias for the feedforward.
        num_register_tokens (int): Number of register tokens.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        gated_mlp (bool): If True, makes the feedforward gated (e.g., for SwiGLU)
        qk_norm (bool): If True, normalizes the query and keys (as in ViT-22B)
    """

    def __init__(
            self,
            input_modalities: list[str],
            output_modalities: list[str],
            modality_info: dict[str, dict[str, Any]],
            tokenizers: torch.nn.ModuleDict,
            cfg: Any,
            text_tokenizer: Any | None = None,
            decoding_steps: list[int] | int = 1,
            temps: list[float] | float = 1.0,
            top_p: float = 0.8,
            top_k: int = 0,
            timesteps: int = 50,
            dim: int = 768,
            encoder_depth: int = 12,
            decoder_depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            mlp_bias: bool = True,
            num_register_tokens: int = 0,
            act_layer: type[nn.Module] = nn.GELU,
            norm_layer: partial | nn.Module = partial(LayerNorm, eps=1e-6),
            gated_mlp: bool = False,
            qk_norm: bool = False,
    ):
        super().__init__()

        if len(input_modalities) == 0:
            raise ValueError("Input modalities not provided.")
        if len(output_modalities) == 0:
            raise ValueError("Output modalities not provided.")

        self.top_p = top_p
        self.top_k = top_k
        self.timesteps = timesteps
        self.decoding_steps = decoding_steps
        self.temps = temps
        self.modality_info = modality_info

        # Include output modalities in encoder so generated tokens can feed back
        # as conditioning for subsequent target modalities (chained generation),
        # and so that ROAR/MaskGIT iterative refinement (decoding_steps > 1) can
        # self-condition within a single output modality.
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.all_modalities = list(dict.fromkeys(input_modalities + output_modalities))  # modalities appear only once

        self.encoder_embeddings, self.decoder_embeddings = init_enc_dec_embeddings(
            cfg=cfg,
            modality_info=modality_info,
            in_domains=self.all_modalities,
            out_domains=output_modalities,
        )

        self.image_modalities = [
            k for k, v in self.encoder_embeddings.items()
            if isinstance(v, (ImageEncoderEmbedding, ImageTokenEncoderEmbedding))
        ]
        self.output_image_modalities = [
            k for k, v in self.decoder_embeddings.items() if isinstance(v, ImageTokenDecoderEmbedding)
        ]

        # Build MAE model
        mae_model = NILunarFM(
            encoder_embeddings=self.encoder_embeddings,
            decoder_embeddings=self.decoder_embeddings,
            modality_info=modality_info,
            dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mlp_bias=mlp_bias,
            num_register_tokens=num_register_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
            gated_mlp=gated_mlp,
            qk_norm=qk_norm,
        )

        self.sampler = GenerationSampler(mae_model, parent=self)

        self.tokenizers = tokenizers
        self.text_tokenizer = text_tokenizer
        self.eos_id = text_tokenizer.token_to_id(EOS_TOKEN) if text_tokenizer is not None else 3
        self.s1_id = text_tokenizer.token_to_id(S1_TOKEN) if text_tokenizer is not None else 36153

    def forward(
            self,
            d: dict[str, torch.Tensor],
            timesteps: int | None = None,
            verbose: bool = False,
            **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            d (dict, torch.Tensor): dict of input tensors
            timesteps (int, optional): Number of diffusion timesteps for generation
            verbose (bool): If True, print generation progress
            **kwargs: Additional keyword arguments with modality=tensor

        Returns:
            dict[str, torch.Tensor]: dict of generated images
        """
        # Check for unknown modalities in input
        for mod in list(d.keys()):
            if mod not in self.all_modalities:
                warnings.warn(f"Unknown input modality: {mod}. Ignoring input.")
                del d[mod]
        if len(d) == 0:
            raise ValueError("No valid inputs provided.")

        batch_size = len(list(d.values())[0])
        device = next(self.parameters()).device

        input_dict = {}
        img_num_tokens, image_size = None, None
        for mod, value in d.items():
            if mod in self.image_modalities:
                # Only run the tokenizer encode path for raw (untokenized) image inputs.
                is_pretokenized = self.modality_info.get(mod, {}).get("pretokenized", False)

                if is_pretokenized:
                    enc_emb = self.encoder_embeddings[mod]
                    img_num_tokens = enc_emb.num_patches
                    image_size = enc_emb.image_size
                else:
                    input_shape = value.shape
                    # Raw image input: shape is (B, C, H, W) after tokenizer encode.
                    if mod in self.tokenizers:
                        value = self.tokenizers[mod].encode(value)
                        if not isinstance(value, dict):
                            value = value[-1]  # Select tokens from img tokenizer

                    patch_size = self.encoder_embeddings[mod].patch_size
                    img_num_tokens = int((input_shape[-1] / patch_size[-1]) * (input_shape[-2] / patch_size[-2]))
                    image_size = (input_shape[-2], input_shape[-1])  # (H, W)

            # Encode input and provide expected format
            input_dict[mod] = init_full_input_modality(
                value, self.modality_info, mod, device, eos_id=self.eos_id, num_tokens=img_num_tokens,
            )

        # Initialize output modalities
        tokens_per_target = []
        autoregression_schemes = []
        token_decoding_schedules = []
        token_decoding_steps = []

        for mod in self.output_modalities:
            if mod in self.output_image_modalities:
                # Get number of tokens from modality_info max_tokens to handle cases where
                # tokenized modality has different patch size than corresponding untokenized
                if self.modality_info[mod].get("max_tokens") is not None:
                    mod_num_tokens = self.modality_info[mod]["max_tokens"]
                else:
                    # Fallback to input token count
                    mod_num_tokens = img_num_tokens
                autoregression_schemes.append("roar")
                token_decoding_schedules.append("linear")
                token_decoding_steps.append(self.decoding_steps)
            else:
                # Get max length from modality info for sequence data
                mod_num_tokens = self.decoder_embeddings[mod].max_length
                autoregression_schemes.append("autoregressive")
                token_decoding_schedules.append(None)
                token_decoding_steps.append(None)
            tokens_per_target.append(mod_num_tokens)

            if mod in input_dict:
                # Modality in input and target
                input_dict[mod] = init_cond_target_modality(
                    input_dict[mod], self.modality_info, mod, mod_num_tokens, eos_id=self.eos_id, s1_id=self.s1_id,
                )
            else:
                input_dict[mod] = init_empty_target_modality(
                    self.modality_info, mod, batch_size, mod_num_tokens, device, s1_id=self.s1_id,
                )

        # Predict tokens of output modalities
        schedule = build_chained_generation_schedules(
            cond_domains=list(d.keys()),
            target_domains=self.output_modalities,
            tokens_per_target=tokens_per_target,
            autoregression_schemes=autoregression_schemes,
            decoding_steps=token_decoding_steps,
            token_decoding_schedules=token_decoding_schedules,
            temps=([self.temps] * len(self.output_modalities) if isinstance(self.temps, (float, int))
                   else list(self.temps) if isinstance(self.temps, list) else [self.temps]),
            temp_schedules=["constant"] * len(self.output_modalities),
            cfg_scales=[1.0] * len(self.output_modalities),
            cfg_schedules=["constant"] * len(self.output_modalities),
            cfg_grow_conditioning=True,
        )

        out_dict = self.sampler.generate(
            input_dict,
            schedule,
            verbose=verbose,
            seed=random.randint(-(2 ** 31), 2 ** 31 - 1),
            top_p=self.top_p,
            top_k=self.top_k,
            text_tokenizer=self.text_tokenizer,
        )

        # TODO Vary timesteps based on codebook diversity
        timesteps = timesteps or self.timesteps
        out = {}
        for mod in self.output_modalities:
            mod_out = out_dict[mod]
            tok = mod_out["tensor"] if isinstance(mod_out, dict) else mod_out
            if mod in self.output_image_modalities:
                if image_size is None:
                    raise ValueError(f"Cannot decode output image modality '{mod}' without image_size. "
                                      "Ensure at least one input modality is an image modality.")
                patch_size = int(self.tokenizers[mod].patch_size)
                nh = image_size[0] // patch_size
                nw = image_size[1] // patch_size
                if tok.ndim == 3:
                    # Multi-codebook: (B, num_tokens, num_codebooks) -> (B, nh, nw, num_codebooks)
                    # FSQ.indices_to_codes with keep_num_codebooks_dim=True and is_img_or_video=True
                    # (ndim>=4) returns (B, D_Q, H_Q, W_Q) as needed by post_quant_proj Conv2d
                    tok = rearrange(tok, "b (nh nw) c -> b nh nw c", nh=nh, nw=nw)
                else:
                    # Single-codebook: (B, num_tokens) -> (B, nh, nw)
                    tok = rearrange(tok, "b (nh nw) -> b nh nw", nh=nh, nw=nw)

                out[mod] = self.tokenizers[mod].decode_tokens(
                    tok,
                    image_size=image_size,
                    timesteps=timesteps,
                    verbose=verbose,
                )

            elif mod in self.output_modalities and mod in ["metadata", "coords", "crater_bboxes", "static_maps"]:
                # Preserve raw generated token ids for sequence modalities to be decoded later instead of per-token.
                # This handles all non-image output modalities (sequences like metadata, static_maps, crater_bboxes, etc.)
                out[mod] = tok

        return out


def get_lfm_generation_model(variant, **kwargs):
    """ Returns a NILunarFM generation model instance based on the specified variant.

    Args:
        variant: A string identifier for the model variant. Supported values include:
            - "tiny": Loads the NILunarFM Tiny model.
            - "small": Loads the NILunarFM Small model.
            - "base": Loads the NILunarFM Base model.
            - "large": Loads the NILunarFM Large model.

        **kwargs: dict - Additional keyword arguments passed to the specific model constructor.

    Returns:
        model: nn.Module - An instance of the selected NILunarFM encoder-decoder model.

    Raises:
        ValueError if the provided variant is not recognized.
    """
    if "base" in variant:
        model = ni_lfm_v1_base_generate(**kwargs)
    elif "large" in variant:
        model = ni_lfm_v1_large_generate(**kwargs)
    elif "small" in variant:
        model = ni_lfm_v1_small_generate(**kwargs)
    elif "tiny" in variant:
        model = ni_lfm_v1_tiny_generate(**kwargs)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    return model


def ni_lfm_v1_tiny_generate(**kwargs):
    """Build NILunarFM v1 tiny model for generation."""
    config = MODEL_CONFIGS["tiny"].copy()
    config.update(kwargs)
    return build_lfm_generate(**config)


def ni_lfm_v1_small_generate(**kwargs):
    """Build NILunarFM v1 small model for generation."""
    config = MODEL_CONFIGS["small"].copy()
    config.update(kwargs)
    return build_lfm_generate(**config)


def ni_lfm_v1_base_generate(**kwargs):
    """Build NILunarFM v1 base model for generation."""
    config = MODEL_CONFIGS["base"].copy()
    config.update(kwargs)
    return build_lfm_generate(**config)


def ni_lfm_v1_large_generate(**kwargs):
    """Build NILunarFM v1 large model for generation."""
    config = MODEL_CONFIGS["large"].copy()
    config.update(kwargs)
    return build_lfm_generate(**config)


def build_lfm_generate(pretrained_tokenizers: torch.nn.ModuleDict, cfg: Any | None, **kwargs):
    """Build NILunarFM generation model with specified configuration.

    Args:
        pretrained_tokenizers: Dictionary of pretrained tokenizer modules
        cfg: model config from pretraining
        **kwargs: Additional model configuration parameters

    Returns:
        NILunarFMGeneration model instance
    """

    model = NILunarFMGeneration(tokenizers=pretrained_tokenizers, cfg=cfg, **kwargs)

    return model


def checkpoint_filter_fn_generate(state_dict: dict, model: NILunarFMGeneration) -> dict:
    """Manually filter pre-trained weights for NILunarFM to enable strict weight loading."""

    model_state_dict = model.state_dict()
    clean_dict = {}
    for k, v in state_dict.items():
        encdec_k = "sampler.model." + k
        if encdec_k in model_state_dict:
            if v.shape == model_state_dict[encdec_k].shape:
                clean_dict[encdec_k] = v
            else:
                print(f"Shape for {k} ({list(v.shape)}) does not match model weights "
                      f"({list(model_state_dict[encdec_k].shape)}), skipping weights.")

    missing_params = set(model_state_dict.keys()) - set(clean_dict.keys())
    tok_keys_preserved = []
    for k in missing_params:
        if k.startswith("tokenizer"):
            # Tokenizer weights are loaded separately; use model state dict to preserve them through strict load.
            tok_keys_preserved.append(k)
        else:
            print(f"Weights for {k} are missing in state dict, using random initialization.")
        clean_dict[k] = model_state_dict[k]

    if tok_keys_preserved:
        print(f"Preserved {len(tok_keys_preserved)} pretrained tokenizer weight tensors through checkpoint filter.")

    state_dict = clean_dict

    return state_dict
