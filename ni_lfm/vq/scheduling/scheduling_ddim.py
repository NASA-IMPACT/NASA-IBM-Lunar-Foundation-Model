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

# DISCLAIMER: This code is strongly influenced by https://github.com/huggingface/diffusers,
# https://github.com/pesser/pytorch_diffusion and https://github.com/hojonathanho/diffusion

import numpy as np
import torch

from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_ddim import DDIMScheduler as DDIMS
from diffusers.schedulers.scheduling_ddim import betas_for_alpha_bar, rescale_zero_terminal_snr

from .scheduling_utils import scaled_cosine_alphas


class DDIMScheduler(DDIMS):
    """ Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
    diffusion probabilistic models (DDPMs) with non-Markovian guidance.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2010.02502

    Args:
        num_train_timesteps (`int`, defaults to 1000): The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001): The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02): The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`): The beta schedule, a mapping from a beta range to a sequence
            of betas for stepping the model. Choose from `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional): Pass an array of betas directly to the constructor to bypass
            `beta_start` and `beta_end`.
        clip_sample (`bool`, default `True`): Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0): The maximum magnitude for sample clipping.
            Valid only when `clip_sample=True`.
        set_alpha_to_one (`bool`, defaults to `True`): Each diffusion step uses the alphas product value at that step
            and at the previous one. For the final step there is no previous alpha. When this option is `True` the
            previous alpha product is fixed to `1`, otherwise it uses the alpha value at step 0.
        steps_offset (`int`, defaults to 0): An offset added to the inference steps. You can use a combination of
            `offset=1` and `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product,
           like in Stable Diffusion.
        prediction_type (`str`, defaults to `epsilon`, optional): Prediction type of the scheduler function; can be
            `epsilon` (predicts the noise of the diffusion process), `sample` (directly predicts the noisy sample`)
            or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)
        thresholding (`bool`, defaults to `False`): Whether to use the "dynamic thresholding" method.
            This is unsuitable for latent-space diffusion models such as Stable Diffusion).
        dynamic_thresholding_ratio (`float`, defaults to 0.995): The ratio for the dynamic thresholding method.
            Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0): The threshold value for dynamic thresholding.
            Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        rescale_betas_zero_snr (`bool`, defaults to `False`): Whether to rescale the betas to have zero terminal SNR.
            This enables the model to generate very bright and dark samples instead of limiting it to samples
            with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: np.ndarray | list[float] | None = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "v_prediction",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "trailing",
        rescale_betas_zero_snr: bool = True,
    ):
        if "shifted_cosine:" in beta_schedule:
            # Syntax is "shifted_cosine:{noise_shift}"
            noise_shift = float(beta_schedule.split(":")[1])
            self.alphas_cumprod = scaled_cosine_alphas(num_train_timesteps, noise_shift)
        else:
            if trained_betas is not None:
                self.betas = torch.tensor(trained_betas, dtype=torch.float32)
            elif beta_schedule == "linear":
                self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                self.betas = (
                    torch.linspace(
                        beta_start**0.5,
                        beta_end**0.5,
                        num_train_timesteps,
                        dtype=torch.float32,
                    )
                    ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                self.betas = betas_for_alpha_bar(num_train_timesteps)
            else:
                raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

            # Rescale for zero SNR
            if rescale_betas_zero_snr:
                self.betas = rescale_zero_terminal_snr(self.betas)

            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def get_alpha_sigma_sqrts(self, timesteps, device, dtype, shape) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=dtype)
        timesteps = timesteps.to(device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod, sqrt_one_minus_alpha_prod

    def get_noise(
        self,
        sample: torch.FloatTensor,
        velocity: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self.get_alpha_sigma_sqrts(
            timesteps, sample.device, sample.dtype, sample.shape
        )
        noise = sqrt_alpha_prod * velocity + sqrt_one_minus_alpha_prod * sample
        return noise
