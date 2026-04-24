"""
Diffusion noise utilities for SubDiff pretraining.
Handles forward diffusion (adding noise) at the patch level.

Two families supported:
- PatchDiffusion: DDPM forward process (x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*eps)
- RectifiedFlow: SD3-style linear interpolation (x_t = (1-t)*x_0 + t*eps)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Rectified Flow (SD3 / FLUX style)
# ---------------------------------------------------------------------------

class RectifiedFlow(nn.Module):
    """Linear interpolation path between data and noise.

    Forward:  x_t = (1-t) * x_0 + t * eps,  t ∈ [0, 1]
    Velocity: v   = dx_t / dt = eps - x_0

    Compared to DDPM:
    - Path is linear (constant velocity), no sqrt curvature
    - Target v is a mixture of ε and x_0 (half-way on manifold)
    - Sampling is straightforward ODE integration
    """

    def __init__(self, t_sampling='logit_normal', logit_mean=0.0, logit_std=1.0):
        super().__init__()
        self.t_sampling = t_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def sample_t(self, batch_size, device):
        """Sample t ∈ (0, 1) per sample.
        - 'uniform': t ~ U(0, 1)
        - 'logit_normal' (SD3): u ~ N(mean, std^2); t = sigmoid(u)
          Concentrates samples around the middle-t regime where structure forms.
        """
        if self.t_sampling == 'uniform':
            t = torch.rand(batch_size, device=device)
        else:  # logit_normal
            u = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            t = torch.sigmoid(u)
        return t.clamp(min=1e-5, max=1.0 - 1e-5)

    def add_noise(self, x_0, t, eps=None):
        """x_t = (1-t) * x_0 + t * eps. Returns (x_t, v, eps) where v = eps - x_0.

        Args:
            x_0: (B, ...) clean samples
            t: (B,) in (0, 1)
        """
        if eps is None:
            eps = torch.randn_like(x_0)
        # Broadcast t to x_0's shape
        shape = [t.shape[0]] + [1] * (x_0.dim() - 1)
        t_b = t.view(*shape)
        x_t = (1 - t_b) * x_0 + t_b * eps
        v = eps - x_0
        return x_t, v, eps


# ---------------------------------------------------------------------------
# DDPM (original)
# ---------------------------------------------------------------------------


def linear_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps, s=0.008):
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999).float()


class PatchDiffusion(nn.Module):
    """
    Applies forward diffusion at the patch level.

    Given an image split into patches, adds noise to a subset of patches
    at a sampled timestep t, while keeping the rest clean.
    """

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 schedule_type='linear'):
        super().__init__()
        self.num_timesteps = num_timesteps

        if schedule_type == 'linear':
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule_type == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def add_noise(self, x_patches, t):
        """
        Add noise to patches using the forward diffusion process: q(x_t | x_0).

        Args:
            x_patches: (B, N, patch_dim) clean patch pixels
            t: (B,) integer timestep for each sample

        Returns:
            noisy_patches: (B, N, patch_dim)
            noise: (B, N, patch_dim) the sampled noise
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]        # (B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        noise = torch.randn_like(x_patches)
        noisy_patches = sqrt_alpha * x_patches + sqrt_one_minus_alpha * noise
        return noisy_patches, noise

    def apply_patch_noise(self, x_patches, noisy_mask, t):
        """
        Selectively add noise to patches based on mask.

        Args:
            x_patches: (B, N, patch_dim) all clean patch pixels
            noisy_mask: (B, N) boolean mask, True = add noise, False = keep clean
            t: (B,) timestep per sample

        Returns:
            mixed_patches: (B, N, patch_dim) mix of noisy and clean patches
            noise: (B, N, patch_dim) noise applied (zero for clean patches)
            noisy_mask: (B, N) the mask used
        """
        noisy_patches, noise = self.add_noise(x_patches, t)

        # Mix: use noisy where mask=True, clean where mask=False
        mask_expanded = noisy_mask.unsqueeze(-1).float()  # (B, N, 1)
        mixed_patches = noisy_patches * mask_expanded + x_patches * (1 - mask_expanded)
        noise = noise * mask_expanded  # zero out noise for clean patches

        return mixed_patches, noise, noisy_mask

    def sample_timesteps(self, batch_size, t_min, t_max, device):
        """
        Sample random timesteps in [t_min, t_max] for each sample in the batch.

        Args:
            batch_size: number of samples
            t_min: minimum timestep (inclusive)
            t_max: maximum timestep (inclusive)
            device: torch device

        Returns:
            t: (B,) integer timesteps
        """
        # Clamp to valid range [0, num_timesteps - 1]
        t_min_clamped = max(int(t_min), 0)
        t_max_clamped = min(int(t_max), self.num_timesteps - 1)
        t = torch.randint(t_min_clamped, t_max_clamped + 1, (batch_size,), device=device)
        return t

    def generate_noisy_mask(self, batch_size, num_patches, clean_ratio, device):
        """
        Generate a random mask indicating which patches are noisy.

        Args:
            batch_size: number of samples
            num_patches: total number of patches per image
            clean_ratio: fraction of patches that should remain clean
            device: torch device

        Returns:
            noisy_mask: (B, N) boolean tensor, True = noisy, False = clean
        """
        num_clean = int(num_patches * clean_ratio)
        # For each image, randomly select clean patch indices
        noisy_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
        for i in range(batch_size):
            clean_indices = torch.randperm(num_patches, device=device)[:num_clean]
            noisy_mask[i, clean_indices] = False
        return noisy_mask
