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
                 total_epochs=300, curriculum_cfg=None,
                 predict_noise=False,
                 mae_masking=False, mask_ratio=0.25,
                 dual_decoder=False, clean_ratio=0.25, pixel_loss_weight=1.0):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2 * in_channels

        # Encoder
        self.encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )

        # Decoder (primary: noise prediction if dual_decoder, else single-task)
        self.decoder = Decoder(
            patch_size=patch_size, num_patches=self.num_patches,
            encoder_dim=embed_dim, decoder_dim=decoder_dim,
            depth=decoder_depth, num_heads=decoder_num_heads
        )

        # Optional second decoder for pixel reconstruction (dual-objective pretraining)
        if dual_decoder:
            self.decoder_pix = Decoder(
                patch_size=patch_size, num_patches=self.num_patches,
                encoder_dim=embed_dim, decoder_dim=decoder_dim,
                depth=decoder_depth, num_heads=decoder_num_heads
            )
        else:
            self.decoder_pix = None

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

        self.predict_noise = predict_noise
        self.mae_masking = mae_masking
        self.mask_ratio = mask_ratio
        self.dual_decoder = dual_decoder
        self.clean_ratio = clean_ratio
        self.pixel_loss_weight = pixel_loss_weight

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
        if self.dual_decoder:
            return self._forward_dual(imgs, epoch)
        if self.mae_masking:
            return self._forward_mae(imgs, epoch)

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
        pred = self.decoder(patch_tokens)  # (B, N, patch_dim)

        if self.predict_noise:
            # Noise prediction: target is the noise added to patches
            # For clean patches, noise is zero (already zeroed in apply_patch_noise)
            target = noise
        else:
            # Pixel reconstruction: target is the clean patches
            target = target_patches

        # Primary loss on noisy regions
        noisy_loss = self._masked_mse(pred, target, noisy_mask)

        # Secondary loss on clean regions
        clean_mask = ~noisy_mask
        clean_loss = self._masked_mse(pred, target, clean_mask)

        loss = noisy_loss + 0.1 * clean_loss

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': noisy_loss.detach(),
            'clean_loss': clean_loss.detach(),
            't_min': t_min,
            't_max': t_max,
            'clean_ratio': clean_ratio,
            't_mean': t.float().mean().detach(),
        }

        return loss, log_dict

    def _forward_mae(self, imgs, epoch=0):
        """MAE-style masked noise prediction (MaskDiT-inspired).

        - Sample per-image timestep t ~ Uniform[0, T) (standard DDPM)
        - Add noise to ALL patches via standard forward diffusion
        - Randomly mask (mask_ratio) of patches; encoder only sees the rest
        - Decoder fills mask tokens and predicts noise for all patches
        - Loss: noise prediction MSE on VISIBLE patches (standard DDPM target)
        - Auxiliary: noise prediction MSE on MASKED patches (MaskDiT regularizer)
        """
        B = imgs.shape[0]
        device = imgs.device

        # 1. Patchify and apply standard forward diffusion to ALL patches
        target_patches = self.patchify(imgs)  # (B, N, patch_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        noisy_patches, noise = self.diffusion.add_noise(target_patches, t)

        # 2. Reconstruct noisy image (all patches noised) for encoder input
        img_size = int(self.patch_size * (self.num_patches ** 0.5))
        noisy_imgs = self.unpatchify(noisy_patches, img_size=img_size)

        # 3. MAE-style masked encoder: only (1 - mask_ratio) visible patches
        cls_token, visible_tokens, ids_restore, mask = self.encoder.forward_masked(
            noisy_imgs, mask_ratio=self.mask_ratio
        )
        # mask: (B, N) with 1 = masked, 0 = visible

        # 4. Decoder: fill mask tokens, predict noise for all patches
        pred = self.decoder.forward_masked(visible_tokens, ids_restore)  # (B, N, patch_dim)

        # 5. Loss: noise prediction. Primary on visible, aux on masked.
        target = noise if self.predict_noise else target_patches
        visible_mask = (mask == 0)  # True where visible
        masked_mask = (mask == 1)   # True where masked

        visible_loss = self._masked_mse(pred, target, visible_mask)
        masked_loss = self._masked_mse(pred, target, masked_mask)

        # MaskDiT weighting: primary on visible, small weight on masked
        loss = visible_loss + 0.1 * masked_loss

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': visible_loss.detach(),
            'clean_loss': masked_loss.detach(),
            't_min': 0,
            't_max': self.diffusion.num_timesteps,
            'clean_ratio': self.mask_ratio,
            't_mean': t.float().mean().detach(),
        }
        return loss, log_dict

    def _forward_dual(self, imgs, epoch=0):
        """Dual-decoder pretraining: shared encoder + two heads.

        Design:
          - Encoder sees clean(clean_ratio) + noisy(1 - clean_ratio) patches
            (no MAE masking — clean patches serve as spatial anchors)
          - t ~ Uniform[0, T) per image (standard DDPM, no curriculum)
          - decoder      → predicts noise ε for each patch (DDPM target)
          - decoder_pix  → reconstructs clean pixel values (MAE target)
          - Loss on the NOISY patches only (clean patches are trivial for both heads)
          - Combined: L = L_eps + pixel_loss_weight * L_pix
        """
        B = imgs.shape[0]
        device = imgs.device

        target_patches = self.patchify(imgs)  # (B, N, patch_dim)

        # Standard DDPM: per-image full-range t
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        # clean_ratio fraction stays clean (spatial anchors), rest is noisy
        noisy_mask = self.diffusion.generate_noisy_mask(
            B, self.num_patches, self.clean_ratio, device
        )
        mixed_patches, noise, noisy_mask = self.diffusion.apply_patch_noise(
            target_patches, noisy_mask, t
        )

        img_size = int(self.patch_size * (self.num_patches ** 0.5))
        mixed_imgs = self.unpatchify(mixed_patches, img_size=img_size)

        # Shared encoder sees 25% clean + 75% noisy patches
        cls_token, patch_tokens = self.encoder(mixed_imgs)

        # Two decoders predict different targets
        pred_eps = self.decoder(patch_tokens)         # noise prediction
        pred_pix = self.decoder_pix(patch_tokens)     # pixel reconstruction

        # Loss on the 75% noisy patches
        loss_eps = self._masked_mse(pred_eps, noise, noisy_mask)
        loss_pix = self._masked_mse(pred_pix, target_patches, noisy_mask)

        loss = loss_eps + self.pixel_loss_weight * loss_pix

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': loss_eps.detach(),   # eps loss reported as "noisy_loss" for logger compat
            'clean_loss': loss_pix.detach(),   # pixel loss reported as "clean_loss"
            't_min': 0,
            't_max': self.diffusion.num_timesteps,
            'clean_ratio': self.clean_ratio,
            't_mean': t.float().mean().detach(),
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
