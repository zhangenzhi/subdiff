"""
SubDiff: Sub-image Patch Diffusion Pretraining.

Main model that combines ViT encoder/decoder with patch-level diffusion
and curriculum learning.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .vit import ViTEncoder, Decoder
from .diffusion import PatchDiffusion
from .curriculum import CurriculumScheduler


class SubDiff(nn.Module):
    """
    SubDiff pretraining model.

    Input: clean image
    Process:
      1. Split image into patches
      2. According to curriculum, select clean/noisy patches and noise strength
      3. Add noise to selected patches
      4. Feed all patches (noisy + clean) into ViT encoder
      5. Decode to reconstruct clean patches
      6. Compute loss on noisy patch regions (and optionally all patches)

    The curriculum gradually reduces noise strength and clean patch ratio,
    transitioning from a MAE-like task (strong noise ≈ mask) to a fine
    denoising task.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_dim=512, decoder_depth=4, decoder_num_heads=8,
                 num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 schedule_type='linear',
                 total_epochs=300, curriculum_cfg=None):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2 * in_channels

        # Encoder
        self.encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )

        # Decoder
        self.decoder = Decoder(
            patch_size=patch_size, num_patches=self.num_patches,
            encoder_dim=embed_dim, decoder_dim=decoder_dim,
            depth=decoder_depth, num_heads=decoder_num_heads
        )

        # Diffusion
        self.diffusion = PatchDiffusion(
            num_timesteps=num_timesteps, beta_start=beta_start,
            beta_end=beta_end, schedule_type=schedule_type
        )

        # Curriculum
        if curriculum_cfg is None:
            curriculum_cfg = {}
        self.curriculum = CurriculumScheduler(
            total_epochs=total_epochs, **curriculum_cfg
        )

    def patchify(self, imgs):
        """Convert images to patch sequences.

        Args:
            imgs: (B, C, H, W)
        Returns:
            patches: (B, N, patch_dim)
        """
        p = self.patch_size
        patches = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                            p1=p, p2=p)
        return patches

    def unpatchify(self, patches, img_size=224):
        """Convert patch sequences back to images.

        Args:
            patches: (B, N, patch_dim)
        Returns:
            imgs: (B, C, H, W)
        """
        p = self.patch_size
        h = w = img_size // p
        imgs = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=h, w=w, p1=p, p2=p, c=3)
        return imgs

    def forward(self, imgs, epoch=0):
        """
        Args:
            imgs: (B, C, H, W) clean images
            epoch: current epoch for curriculum scheduling

        Returns:
            loss: reconstruction loss
            log_dict: dict with metrics for logging
        """
        B = imgs.shape[0]
        device = imgs.device

        # Get curriculum parameters
        curriculum_state = self.curriculum.get_state(epoch)
        t_min, t_max = curriculum_state['t_min'], curriculum_state['t_max']
        clean_ratio = curriculum_state['clean_ratio']

        # Patchify
        target_patches = self.patchify(imgs)  # (B, N, patch_dim)

        # Sample timesteps and generate noisy mask
        t = self.diffusion.sample_timesteps(B, t_min, t_max, device)
        noisy_mask = self.diffusion.generate_noisy_mask(
            B, self.num_patches, clean_ratio, device
        )

        # Apply noise to selected patches
        mixed_patches, noise, noisy_mask = self.diffusion.apply_patch_noise(
            target_patches, noisy_mask, t
        )

        # Reconstruct image from mixed patches for encoder input
        mixed_imgs = self.unpatchify(mixed_patches, img_size=int(self.patch_size * (self.num_patches ** 0.5)))

        # Encode
        cls_token, patch_tokens = self.encoder(mixed_imgs)

        # Decode
        pred_patches = self.decoder(patch_tokens)  # (B, N, patch_dim)

        # Loss: MSE on noisy patches (primary), optionally also on clean patches
        # Primary loss on noisy regions
        noisy_loss = self._masked_mse(pred_patches, target_patches, noisy_mask)

        # Secondary loss on clean regions (much smaller weight, regularization)
        clean_mask = ~noisy_mask
        clean_loss = self._masked_mse(pred_patches, target_patches, clean_mask)

        loss = noisy_loss + 0.1 * clean_loss

        log_dict = {
            'loss': loss.item(),
            'noisy_loss': noisy_loss.item(),
            'clean_loss': clean_loss.item(),
            't_min': t_min,
            't_max': t_max,
            'clean_ratio': clean_ratio,
            't_mean': t.float().mean().item(),
        }

        return loss, log_dict

    def _masked_mse(self, pred, target, mask):
        """Compute MSE loss only on masked (selected) patches."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = (pred - target) ** 2
        diff = diff.mean(dim=-1)  # (B, N) mean over patch_dim
        loss = (diff * mask.float()).sum() / mask.float().sum()
        return loss

    def get_encoder(self):
        """Return the encoder for downstream evaluation."""
        return self.encoder
