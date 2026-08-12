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

import numpy as np
import torch


def scaled_cosine_alphas(num_diffusion_timesteps: int, noise_shift: float = 1.0) -> torch.Tensor:
    """Shifts a cosine noise schedule by a specified amount in log-SNR space.

    noise_shift = 1.0 corresponds to the standard cosine noise schedule.
    0 < noise_shift < 1.0 corresponds to a less noisy schedule (better
    suited if the conditioning is highly informative, e.g. low-res images).
    noise_shift > 1.0 corresponds to a more noisy schedule (better suited
    if the conditioning is not as informative, e.g. captions).

    See https://arxiv.org/abs/2305.18231

    Args:
        num_diffusion_timesteps: the number of diffusion timesteps.
        noise_shift: the amount to shift the noise schedule by in log-SNR space.

    Returns:
        The alphas_cumprod used by the diffusion noise scheduler
    """
    t = torch.linspace(0, 1, num_diffusion_timesteps).to(torch.float64)
    log_snr = -2 * (torch.tan(torch.pi * t / 2).log() + np.log(noise_shift))
    log_snr = log_snr.clamp(-15, 15).float()
    alphas_cumprod = log_snr.sigmoid()
    alphas_cumprod[-1] = 0.0
    return alphas_cumprod
