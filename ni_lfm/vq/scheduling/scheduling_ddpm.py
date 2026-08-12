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

# DISCLAIMER: This code is strongly influenced by https://github.com/huggingface/diffusers
# and https://github.com/ermongroup/ddim

import numpy as np
import torch

from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_ddim import rescale_zero_terminal_snr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as DDPMS
from diffusers.schedulers.scheduling_ddpm import betas_for_alpha_bar

from .scheduling_utils import scaled_cosine_alphas


class DDPMScheduler(DDPMS):
    """ Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
    Langevin dynamics sampling.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2006.11239

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): The starting `beta` value of inference.
        beta_end (`float`): The final `beta` value.
        beta_schedule (`str`): The beta schedule, a mapping from a beta range to a sequence of betas
            for stepping the model. Choose from `linear`, `scaled_linear`, `squaredcos_cap_v2` or `sigmoid`.
        trained_betas (`np.ndarray`, optional): Option to pass an array of betas directly to the
            constructor to bypass `beta_start`, `beta_end` etc.
        variance_type (`str`): Options to clip the variance used when adding noise to the denoised sample.
            Choose from `fixed_small`,`fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, default `True`): Option to clip predicted sample for numerical stability.
        clip_sample_range (`float`, default `1.0`): The maximum magnitude for sample clipping.
            Valid only when `clip_sample=True`.
        prediction_type (`str`, default `epsilon`, optional): Prediction type of the scheduler function, one of
            `epsilon` (predicting the noise of the diffusion process), `sample` (directly predicting the noisy sample`)
            or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)
        thresholding (`bool`, default `False`): Whether to use the "dynamic thresholding" method
            (introduced by Imagen, https://arxiv.org/abs/2205.11487). Note that the thresholding method
            is unsuitable for latent-space diffusion models (such as stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`): The ratio for the dynamic thresholding method. Default
            is `0.995`, the same as Imagen (https://arxiv.org/abs/2205.11487). Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to `1.0`): The threshold value for dynamic thresholding.
            Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`): The way the timesteps should be scaled. Refer to Table 2 of
            the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)
            for more information.
        steps_offset (`int`, defaults to 0): An offset added to the inference steps. You can use a combination of
            `offset=1` and `set_alpha_to_one=False` to make the last step use step 0 for the previous alpha product
            like in Stable Diffusion.
        zero_terminal_snr: bool
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: np.ndarray | list[float] | None = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "v_prediction",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
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
                    torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                self.betas = betas_for_alpha_bar(num_train_timesteps)
            elif beta_schedule == "sigmoid":
                # GeoDiff sigmoid schedule
                betas = torch.linspace(-6, 6, num_train_timesteps)
                self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
            else:
                raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

            if rescale_betas_zero_snr:
                self.betas = rescale_zero_terminal_snr(self.betas)

            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type

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
