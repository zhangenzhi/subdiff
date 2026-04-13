"""
Diffusion noise utilities for SubDiff pretraining.
Handles forward diffusion (adding noise) at the patch level.
"""

import math
import torch
import torch.nn as nn


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
        t = torch.randint(int(t_min), int(t_max) + 1, (batch_size,), device=device)
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
