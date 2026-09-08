# Copyright 2024 EPFL and Apple Inc.
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
import copy

from contextlib import nullcontext
from typing import Any

import torch

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin
from vector_quantize_pytorch import FSQ

import ni_lfm.vq.models.unet.unet_diffusion as unet_diffusion
import ni_lfm.vq.models.vit_models as vit_models

from ni_lfm.vq.scheduling import DDIMScheduler, DDPMScheduler, PipelineCond


# If freeze_enc is True, the following modules will be frozen
FREEZE_MODULES = ["encoder", "quant_proj", "quantize", "cls_emb"]


class VQ(torch.nn.Module, PyTorchModelHubMixin):
    """Base class for VQVAE and DiVAE models. Implements the encoder and quantizer, and can be used as such without
    a decoder after training.

    Args:
        image_size: Input and target image size.
        image_size_enc: Input image size for the encoder. Defaults to image_size. Change this when loading weights
          from a tokenizer trained on a different image size.
        n_channels: Number of input channels.
        n_labels: Number of classes for semantic segmentation.
        enc_type: String identifier specifying the encoder architecture. See vq/vit_models.py
            for available architectures.
        patch_proj: Whether or not to use a ViT-style patch-wise linear projection in the encoder.
        post_mlp: Whether or not to add a small point-wise MLP before the quantizer.
        patch_size: Patch size for the encoder.
        codebook_size: Number of codebook entries.
        num_codebooks: Number of "parallel" codebooks to use.
        latent_dim: Dimensionality of the latent code.
        ckpt_path: Path to a checkpoint to load the model weights from.
        ignore_keys: List of keys to ignore when loading the state_dict from the above checkpoint.
        freeze_enc: Whether or not to freeze the encoder weights. See FREEZE_MODULES for the list of modules.
        config: Dictionary containing the model configuration. Only used when loading
            from Huggingface Hub. Ignore otherwise.
    """

    def __init__(
        self,
        image_size: int = 224,
        image_size_enc: int | None = None,
        n_channels: str = 12,
        n_labels: int | None = None,
        enc_type: str = "vit_b_enc",
        patch_proj: bool = True,
        post_mlp: bool = False,
        patch_size: int = 16,
        codebook_size: int | str = 16384,
        num_codebooks: int = 1,
        latent_dim: int = 32,
        ckpt_path: str | None = None,
        ignore_keys: list[str] = [
            "decoder",
            "loss",
            "post_quant_conv",
            "post_quant_proj",
            "encoder.pos_emb",
        ],
        freeze_enc: bool = False,
        undo_std: bool = False,
        config: dict[str, Any] | None = None,
        **kwargs,
    ):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return

        super().__init__()

        self.image_size = image_size
        self.n_channels = n_channels
        self.n_labels = n_labels
        self.enc_type = enc_type
        self.patch_proj = patch_proj
        self.post_mlp = post_mlp
        self.patch_size = patch_size
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.latent_dim = latent_dim
        self.ckpt_path = ckpt_path
        self.ignore_keys = ignore_keys
        self.freeze_enc = freeze_enc
        self.undo_std = undo_std

        # For semantic segmentation
        if n_labels is not None:
            self.cls_emb = torch.nn.Embedding(num_embeddings=n_labels, embedding_dim=n_channels)
            self.colorize = torch.randn(3, n_labels, 1, 1)
        else:
            self.cls_emb = None

        # Init encoder
        image_size_enc = image_size_enc or image_size
        if "vit" in enc_type:
            self.encoder = getattr(vit_models, enc_type)(
                in_channels=n_channels,
                patch_size=patch_size,
                resolution=image_size_enc,
                patch_proj=patch_proj,
                post_mlp=post_mlp,
            )
            self.enc_dim = self.encoder.dim_tokens
        else:
            raise NotImplementedError(f"{enc_type} not implemented.")

        # Encoder -> quantizer projection
        self.quant_proj = torch.nn.Conv2d(self.enc_dim, self.latent_dim, 1)

        # Init quantizer
        levels = list(map(int, codebook_size.split("-")))
        self.quantize = FSQ(levels=levels, num_codebooks=num_codebooks)

        # Load checkpoint
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # Freeze encoder
        if freeze_enc:
            for module_name, module in self.named_children():
                if module_name not in FREEZE_MODULES:
                    continue
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def train(self, mode: bool = True) -> "VQ":
        """Override the default train function to set the training mode to all modules
        except the encoder if freeze_enc is True.

        Args:
            mode: Whether to set the model to training mode (True) or evaluation mode (False).
        """
        if not isinstance(mode, bool):
            raise TypeError("training mode is expected to be boolean")
        self.training = mode
        for module_name, module in self.named_children():
            if self.freeze_enc and module_name in FREEZE_MODULES:
                continue
            module.train(mode)
        return self

    def init_from_ckpt(self, path: str, ignore_keys: list[str] = list()) -> "VQ":
        """Loads the state_dict from a checkpoint file and initializes the model with it.
        Renames the keys in the state_dict if necessary (e.g. when loading VQ-GAN weights).

        Args:
            path: Path to the checkpoint file.
            ignore_keys: List of keys to ignore when loading the state_dict.

        Returns:
            self
        """
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]

        # Compatibility with ViT-VQGAN weights
        if "quant_conv.0.weight" in sd and "quant_conv.0.bias" in sd:
            print("Renaming quant_conv.0 to quant_proj")
            sd["quant_proj.weight"] = sd["quant_conv.0.weight"]
            sd["quant_proj.bias"] = sd["quant_conv.0.bias"]
            del sd["quant_conv.0.weight"]
            del sd["quant_conv.0.bias"]
        elif "quant_conv.weight" in sd and "quant_conv.bias" in sd:
            print("Renaming quant_conv to quant_proj")
            sd["quant_proj.weight"] = sd["quant_conv.weight"]
            sd["quant_proj.bias"] = sd["quant_conv.bias"]
            del sd["quant_conv.weight"]
            del sd["quant_conv.bias"]
        if "post_quant_conv.0.weight" in sd and "post_quant_conv.0.bias" in sd:
            print("Renaming post_quant_conv.0 to post_quant_proj")
            sd["post_quant_proj.weight"] = sd["post_quant_conv.0.weight"]
            sd["post_quant_proj.bias"] = sd["post_quant_conv.0.bias"]
            del sd["post_quant_conv.0.weight"]
            del sd["post_quant_conv.0.bias"]
        elif "post_quant_conv.weight" in sd and "post_quant_conv.bias" in sd:
            print("Renaming post_quant_conv to post_quant_proj")
            sd["post_quant_proj.weight"] = sd["post_quant_conv.weight"]
            sd["post_quant_proj.bias"] = sd["post_quant_conv.bias"]
            del sd["post_quant_conv.weight"]
            del sd["post_quant_conv.bias"]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        msg = self.load_state_dict(sd, strict=False)
        print(msg)
        print(f"Restored from {path}")

        return self

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocesses the input image tensor before feeding it to the encoder.
        If self.undo_std, the input is first denormalized from the ImageNet
        standardization to [-1, 1]. If semantic segmentation is performed, the
        class indices are embedded.

        Args:
            x: Input image tensor of shape B C H W or B H W in case of semantic segmentation

        Returns:
            Preprocessed input tensor of shape B C H W
        """
        if self.cls_emb is not None:
            x = rearrange(self.cls_emb(x), "b h w c -> b c h w")
        return x

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Encodes an input image tensor and quantizes the latent code.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation

        Returns:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            code_loss: Codebook loss
            tokens: Quantized indices of shape B H_Q W_Q
        """
        x = self.prepare_input(x)
        h = self.encoder(x)
        h = self.quant_proj(h)

        # FSQ quantizer
        quant, tokens = self.quantize(h)
        code_loss = torch.Tensor([0.]).to(h.device)

        return quant, code_loss, tokens

    def tokenize(self, x: torch.Tensor) -> torch.LongTensor:
        """Tokenizes an input image tensor.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation

        Returns:
            Quantized indices of shape B H_Q W_Q
        """
        _, _, tokens = self.encode(x)
        return tokens

    def autoencode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Autoencodes an input image tensor by encoding it, quantizing the latent code,
        and decoding it back to an image.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation

        Returns:
            Reconstructed image tensor of shape B C H W
        """
        pass

    def decode_quant(self, quant: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            quant: Quantized latent code of shape B D_Q H_Q W_Q

        Returns:
            Decoded image tensor of shape B C H W
        """
        pass

    def tokens_to_embedding(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Look up the codebook entries corresponding the discrete tokens.

        Args:
            tokens: Quantized indices of shape B H_Q W_Q (single codebook)
                or B H_Q W_Q C (multi-codebook FSQ)

        Returns:
            Quantized latent code of shape B D_Q H_Q W_Q
        """
        quant = self.quantize.indices_to_codes(tokens)

        # FSQ with num_codebooks > 1 returns (B, H, W, D) — channels last.
        # Conv2d-based decoders (post_quant_proj) expect (B, D, H, W).
        # Detect channels-last: ndim==4 and last dim matches latent_dim but
        # second dim matches the spatial height of the token grid.
        if (quant.ndim == 4
                and quant.shape[-1] == self.latent_dim
                and quant.shape[1] != self.latent_dim):
            quant = quant.permute(0, 3, 1, 2).contiguous()

        return quant

    def decode_tokens(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        """Decodes discrete tokens back to an image.

        Args:
            tokens: Quantized indices of shape B H_Q W_Q

        Returns:
            Decoded image tensor of shape B C H W
        """
        quant = self.tokens_to_embedding(tokens)

        # Get image size from token shape
        image_size = (kwargs.pop("image_size", None) or
                      (tokens.shape[-2] * self.patch_size, tokens.shape[-1] * self.patch_size))

        dec = self.decode_quant(quant, image_size=image_size, **kwargs)
        return dec

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder and quantizer.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation

        Returns:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            code_loss: Codebook loss
        """
        quant, code_loss, _ = self.encode(x)
        return quant, code_loss


class DiVAE(VQ):
    """DiVAE ("Diffusion VQ-VAE") model = simple encoder + diffusion decoder with
    a discrete bottleneck, inspired by https://arxiv.org/abs/2206.00386.

    Args:
        dec_type: String identifier specifying the decoder architecture.
          See vq/models/unet/unet.py and vq/models/uvit.py for available architectures.
        num_train_timesteps: Number of diffusion timesteps to use for training.
        cls_free_guidance_dropout: Dropout probability for classifier-free guidance.
        masked_cfg: Whether or not to randomly mask out conditioning tokens.
          cls_free_guidance_dropout must be > 0.0 for this to have any effect, and
          decides how often masking is performed. E.g. with 0.5, half of the time
          the conditioning tokens will be randomly masked, and half the time they
          will be kept as is.
        masked_cfg_low: Lower bound of number of tokens to mask out.
        masked_cfg_high: Upper bound of number of tokens to mask out (inclusive).
          Defaults to total number of tokens (H_Q * W_Q) if it is set to None.
        scheduler: String identifier specifying the diffusion scheduler to use.
            Can be 'ddpm' or 'ddim'.
        beta_schedule: String identifier specifying the beta schedule to use for
          the diffusion process. Can be 'linear', 'squaredcos_cap_v2' (cosine),
          'shifted_cosine:{shift_amount}'; see vq/scheduling for details.
        prediction_type: String identifier specifying the type of prediction to use.
          Can be 'sample', 'epsilon', or 'v_prediction'; see vq/scheduling for details.
        clip_sample: Whether or not to clip the samples to [-1, 1], at inference only.
        thresholding: Whether or not to use dynamic thresholding  (introduced by Imagen,
          https://arxiv.org/abs/2205.11487) for the diffusion process, at inference only.
        conditioning: String identifier specifying the way to condition the diffusion
          decoder. Can be 'concat' or 'xattn'. See models for details (only relevant to UViT).
        dec_transformer_dropout: Dropout rate for the transformer layers in the
          diffusion decoder (only relevant to UViT models).
        zero_terminal_snr: Whether or not to enforce zero terminal SNR, i.e. the SNR
          at the last timestep is set to zero. This is useful for preventing the model
          from "cheating" by using information in the last timestep to reconstruct the image.
          See https://arxiv.org/abs/2305.08891.
        image_size_dec: Image size for the decoder. Defaults to image_size.
          Change this when loading weights from a tokenizer decoder trained on a
          different image size.
        config: Dictionary containing the model configuration. Only used when loading
            from Huggingface Hub. Ignore otherwise.
    """

    def __init__(
        self,
        dec_type: str = "unet_patched",
        num_train_timesteps: int = 1000,
        cls_free_guidance_dropout: float = 0.0,
        masked_cfg: bool = False,
        masked_cfg_low: int = 0,
        masked_cfg_high: int | None = None,
        scheduler: str = "ddpm",
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "v_prediction",
        clip_sample: bool = False,
        thresholding: bool = True,
        zero_terminal_snr: bool = True,
        image_size_dec: int | None = None,
        config: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return
        # Don't want to load the weights just yet
        self.original_ckpt_path = kwargs.get("ckpt_path")
        kwargs["ckpt_path"] = None
        super().__init__(*args, **kwargs)
        self.ckpt_path = self.original_ckpt_path
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.thresholding = thresholding
        self.zero_terminal_snr = zero_terminal_snr

        if cls_free_guidance_dropout > 0.0:
            self.cfg_dist = torch.distributions.Bernoulli(probs=cls_free_guidance_dropout)
        else:
            self.cfg_dist = None
        self.masked_cfg = masked_cfg
        self.masked_cfg_low = masked_cfg_low
        self.masked_cfg_high = masked_cfg_high

        # Init diffusion decoder
        image_size_dec = image_size_dec or self.image_size
        if "unet_" in dec_type:
            self.decoder = unet_diffusion.unet_patched(
                in_channels=self.n_channels,
                out_channels=self.n_channels,
                cond_channels=self.latent_dim,
                image_size=image_size_dec,
            )
        else:
            raise NotImplementedError(f"dec_type {dec_type} not implemented.")

        # Init training diffusion scheduler / default pipeline for generation
        scheduler_cls = DDPMScheduler if scheduler == "ddpm" else DDIMScheduler
        self.noise_scheduler = scheduler_cls(
            num_train_timesteps=num_train_timesteps,
            thresholding=thresholding,
            clip_sample=clip_sample,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=zero_terminal_snr,
        )
        self.pipeline = PipelineCond(
            model=self.decoder,
            scheduler=self.noise_scheduler,
            n_channels=self.n_channels,
        )
        # Load checkpoint
        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path, ignore_keys=self.ignore_keys)

    def sample_mask(self, quant: torch.Tensor, low: int = 0, high: int | None = None) -> torch.BoolTensor:
        """Returns a mask of shape B H_Q W_Q, where True = masked-out, False = keep.

        Args:
            quant: Dequantized latent tensor of shape B D_Q H_Q W_Q
            low: Lower bound of number of tokens to mask out
            high: Upper bound of number of tokens to mask out (inclusive).
              Defaults to total number of tokens (H_Q * W_Q) if it is set to None.

        Returns:
            Boolean mask of shape B H_Q W_Q
        """
        B, _, H_Q, W_Q = quant.shape
        num_tokens = H_Q * W_Q
        high = high if high is not None else num_tokens

        zero_idxs = torch.randint(low=low, high=high + 1, size=(B,), device=quant.device)
        noise = torch.rand(B, num_tokens, device=quant.device)
        ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        mask = torch.where(ids_arange_shuffle < zero_idxs.unsqueeze(1), 0, 1)
        mask = rearrange(mask, "b (h w) -> b h w", h=H_Q, w=W_Q).bool()

        return mask

    def _get_pipeline(self, scheduler: SchedulerMixin | None = None) -> PipelineCond:
        """Creates a conditional diffusion pipeline with the given scheduler.

        Args:
            scheduler: Scheduler to use for the diffusion pipeline.
              If None, the default scheduler will be used.

        Returns:
            Conditional diffusion pipeline.
        """

        if scheduler is None:
            return self.pipeline

        return PipelineCond(model=self.decoder, scheduler=scheduler, n_channels=self.n_channels)

    def decode_quant(
        self,
        quant: torch.Tensor,
        timesteps: int | None = None,
        scheduler: SchedulerMixin | None = None,
        generator: torch.Generator | None = None,
        image_size: tuple[int, int] | int | None = None,
        verbose: bool = False,
        orig_res: torch.LongTensor | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            timesteps: Number of diffusion timesteps to use. Defaults to self.num_train_timesteps.
            scheduler: Scheduler to use for the diffusion pipeline. Defaults to the training scheduler.
            generator: Random number generator to use for sampling. By default generations are stochastic.
            image_size: Image size to use for the diffusion pipeline. Defaults to decoder image size.
            verbose: Whether or not to print progress bar.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.

        Returns:
            Decoded image tensor of shape B C H W
        """
        pipeline = self._get_pipeline(scheduler)
        dec = pipeline(
            quant,
            timesteps=timesteps,
            generator=generator,
            image_size=image_size,
            verbose=verbose,
            orig_res=orig_res,
        )
        return dec

    def decode_tokens(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        """See `decode_quant` for details on the optional args."""
        return super().decode_tokens(tokens, **kwargs)

    def autoencode(
        self,
        input_clean: torch.Tensor,
        timesteps: int | None = None,
        scheduler: SchedulerMixin | None = None,
        generator: torch.Generator | None = None,
        verbose: bool = True,
        orig_res: torch.LongTensor | tuple[int, int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoencodes an input image tensor by encoding it, quantizing the latent code,
            and decoding it back to an image.

        Args:
            input_clean: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
            timesteps: Number of diffusion timesteps to use. Defaults to self.num_train_timesteps.
            scheduler: Scheduler to use for the diffusion pipeline. Defaults to the training scheduler.
            generator: Random number generator to use for sampling. By default generations are stochastic.
            verbose: Whether or not to print progress bar.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.

        Returns:
            Reconstructed image tensor of shape B C H W
        """
        pipeline = self._get_pipeline(scheduler)
        quant, _, _ = self.encode(input_clean)
        image_size = input_clean.shape[-1]
        dec = pipeline(
            quant,
            timesteps=timesteps,
            generator=generator,
            image_size=image_size,
            verbose=verbose,
            orig_res=orig_res,
        )
        return dec

    def forward(
        self,
        input_clean: torch.Tensor,
        input_noised: torch.Tensor,
        timesteps: torch.Tensor | float | int,
        cond_mask: torch.Tensor | None = None,
        orig_res: torch.LongTensor | tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder, quantizer, and decoder.

        Args:
            input_clean: Clean input image tensor of shape B C H W
              or B H W in case of semantic segmentation. Used for encoding.
            input_noised: Noised input image tensor of shape B C H W. Used as
              input to the diffusion decoder.
            timesteps: Timesteps for conditioning the diffusion decoder on.
            cond_mask: Optional mask for the diffusion conditioning.
              True = masked-out, False = keep.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.

        Returns:
            dec: Decoded image tensor of shape B C H W
            code_loss: Codebook loss
        """
        with torch.no_grad() if self.freeze_enc else nullcontext():
            quant, code_loss, _ = self.encode(input_clean)

        if cond_mask is None and self.cfg_dist is not None and self.training:
            # Create a random mask for each batch element. True = masked-out, False = keep
            B, _, H_Q, W_Q = quant.shape
            cond_mask = self.cfg_dist.sample((B,)).to(quant.device, dtype=torch.bool)
            cond_mask = repeat(cond_mask, "b -> b h w", h=H_Q, w=W_Q)
            if self.masked_cfg:
                mask = self.sample_mask(quant, low=self.masked_cfg_low, high=self.masked_cfg_high)
                cond_mask = mask * cond_mask

        dec = self.decoder(input_noised, timesteps, quant, cond_mask=cond_mask, orig_res=orig_res)
        return dec, code_loss


def build_divae(cfg, image_size: int) -> DiVAE:
    """Creates DiVAE based on config *cfg*.

    Model hyperparameters are fixed across all trained tokenizers, so only
    `image_size` and `n_channels` are read from `cfg`.
    """
    # Support both original (nested by domain) and slim (flat) stats layouts.
    stats = cfg.data.stats
    channels = stats["channels"] if "channels" in stats else stats[cfg.data.domain]["channels"]
    n_channels = len(channels)

    return DiVAE(
        image_size=image_size,
        n_channels=n_channels,
        patch_size=16,
        enc_type="vit_b_enc",
        dec_type="unet_patched",
        post_mlp=True,
        codebook_size="8-8-8-6-5",
        num_codebooks=1,
        latent_dim=5,
        num_train_timesteps=1000,
        scheduler="ddpm",
        beta_schedule="linear",
        prediction_type="sample",
        zero_terminal_snr=True,
        thresholding=False,
        clip_sample=False,
    )
