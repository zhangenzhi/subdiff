"""
SubDiff: Sub-image Patch Diffusion Pretraining.

Main model that combines ViT encoder/decoder with patch-level diffusion
and curriculum learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .vit import ViTEncoder, DiTEncoder, Decoder
from .diffusion import PatchDiffusion, RectifiedFlow
from .curriculum import CurriculumScheduler


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP, as in DDPM/DiT."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


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
                 dual_decoder=False, clean_ratio=0.25, pixel_loss_weight=1.0,
                 naive_mae=False,
                 naive_ddpm=False,
                 qk_norm=False,
                 dit_minimal_head=False,
                 use_indicators=False,
                 use_conv_refine=False,
                 loss_weighting='simple', snr_gamma=5.0,
                 pos_embed_type='sincos',
                 flow_matching=False, rf_t_sampling='logit_normal',
                 rf_logit_mean=0.0, rf_logit_std=1.0,
                 rf_mae_enabled=False, rf_mae_max_mask=0.5,
                 mae_aux_weight=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2 * in_channels

        # Encoder selection:
        #   - Any diffusion variant (naive_ddpm, predict_noise, dual_decoder)
        #     uses DiTEncoder with per-block adaLN-Zero time conditioning
        #     (prevents time-signal dilution through the 12-layer stack).
        #   - Plain pixel-reconstruction pretraining uses ViTEncoder.
        use_dit_encoder = naive_ddpm or predict_noise or dual_decoder
        if use_dit_encoder:
            self.encoder = DiTEncoder(
                img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                qk_norm=qk_norm,
                pos_embed_type=pos_embed_type,
            )
        else:
            self.encoder = ViTEncoder(
                img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                qk_norm=qk_norm,
                pos_embed_type=pos_embed_type,
            )
        self._use_dit_encoder = use_dit_encoder

        # Decoder selection:
        #  - dit_minimal_head: DiT-style single Linear head, zero-initialized.
        #    Skips the 4-layer external Decoder. Works for naive_ddpm AND
        #    SubDiff variants. The 4-layer Decoder without time conditioning
        #    is what destroyed time-conditioned representations from encoder,
        #    breaking sampling. Replacing with single Linear restores it.
        #  - else: 4-layer Decoder transformer (default).
        self.dit_minimal_head = dit_minimal_head
        if self.dit_minimal_head:
            self.decoder = nn.Linear(embed_dim, self.patch_dim, bias=True)
            nn.init.zeros_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
        else:
            self.decoder = Decoder(
                patch_size=patch_size, num_patches=self.num_patches,
                encoder_dim=embed_dim, decoder_dim=decoder_dim,
                depth=decoder_depth, num_heads=decoder_num_heads,
                qk_norm=qk_norm,
            )

        # Optional second decoder for pixel reconstruction (dual-objective pretraining)
        if dual_decoder:
            if self.dit_minimal_head:
                self.decoder_pix = nn.Linear(embed_dim, self.patch_dim, bias=True)
                nn.init.zeros_(self.decoder_pix.weight)
                nn.init.zeros_(self.decoder_pix.bias)
            else:
                self.decoder_pix = Decoder(
                    patch_size=patch_size, num_patches=self.num_patches,
                    encoder_dim=embed_dim, decoder_dim=decoder_dim,
                    depth=decoder_depth, num_heads=decoder_num_heads,
                    qk_norm=qk_norm,
                )
        else:
            self.decoder_pix = None

        # Diffusion
        self.diffusion = PatchDiffusion(
            num_timesteps=num_timesteps, beta_start=beta_start,
            beta_end=beta_end, schedule_type=schedule_type
        )

        # Optional Rectified Flow (SD3 / FLUX style). When enabled, replaces
        # the DDPM forward path with linear interpolation and v-prediction.
        self.flow_matching = flow_matching
        if flow_matching:
            self.rf = RectifiedFlow(
                t_sampling=rf_t_sampling,
                logit_mean=rf_logit_mean,
                logit_std=rf_logit_std,
            )
        else:
            self.rf = None

        # Curriculum
        if curriculum_cfg is None:
            curriculum_cfg = {}
        self.curriculum = CurriculumScheduler(
            total_epochs=total_epochs, **curriculum_cfg
        )

        # Optional indicator embeddings.
        #   - use_indicators=True: adds learnable noise/clean indicator to
        #     each patch, telling encoder explicitly which are anchors.
        #     Creates a training-inference mismatch (at sampling, all patches
        #     are noisy so all get noise_indicator).
        #   - use_indicators=False (default): encoder must learn clean vs
        #     noisy from content alone. Matches inference distribution.
        self.use_indicators = use_indicators
        if use_indicators:
            self.noise_indicator = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.clean_indicator = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.noise_indicator, std=0.02)
            nn.init.trunc_normal_(self.clean_indicator, std=0.02)

        # Loss weighting scheme (applies to ε prediction paths).
        #  - 'simple': uniform weighting (standard DDPM, what we've been using)
        #  - 'min_snr': Min-SNR-γ (Hang et al. 2023). Weight per sample
        #    = min(SNR(t), gamma). Downweights the high-t regime where the
        #    model can only predict the dataset mean (mode collapse attractor),
        #    and caps low-t weighting at gamma to avoid fine-detail over-focus.
        #    Used in SD3 / PixArt-Σ / FLUX.
        self.loss_weighting = loss_weighting
        self.snr_gamma = snr_gamma

        # Optional Conv refinement head: small residual Conv stack applied
        # after unpatchify, to smooth patch boundaries. Addresses the
        # "patch-independent generation" failure mode of pure ViT + Linear head
        # in pixel space (what VAE's Conv decoder does implicitly in latent DiT).
        # Final Conv is zero-initialized so refinement starts as identity.
        self.use_conv_refine = use_conv_refine
        if use_conv_refine:
            self.conv_refine = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.conv_refine[-1].weight)
            nn.init.zeros_(self.conv_refine[-1].bias)
        else:
            self.conv_refine = None

        self.predict_noise = predict_noise
        self.mae_masking = mae_masking
        self.mask_ratio = mask_ratio
        self.dual_decoder = dual_decoder
        self.clean_ratio = clean_ratio
        self.pixel_loss_weight = pixel_loss_weight
        self.naive_mae = naive_mae
        self.naive_ddpm = naive_ddpm

        # RF + MAE mask (MaskDiT-style): replace a fraction of patch tokens
        # with a learnable mask_token embedding before the encoder, forcing
        # cross-patch reasoning. All tokens go through encoder (no asymmetric
        # encoder-decoder); target is still v on all tokens.
        self.rf_mae_enabled = rf_mae_enabled
        self.rf_mae_max_mask = rf_mae_max_mask
        self.mae_aux_weight = mae_aux_weight
        if rf_mae_enabled:
            assert flow_matching and naive_ddpm, \
                "rf_mae_enabled requires flow_matching=True and naive_ddpm=True"
            self.rf_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.rf_mask_token, std=0.02)
        else:
            self.rf_mask_token = None

        # Time embedding: used by naive_ddpm (adaLN per block) and by
        # SubDiff eps-prediction paths (additive to patch tokens).
        # For pixel-only prediction without predict_noise, t is irrelevant
        # so we skip creating time_embed to avoid unused params.
        if naive_ddpm or predict_noise or dual_decoder:
            self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        else:
            self.time_embed = None

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

    def _eps_weight(self, t):
        """Per-sample loss weight for ε prediction based on self.loss_weighting.

        Args:
            t: (B,) integer timesteps
        Returns:
            (B,) weights, already normalized so mean ≈ 1 on uniform t (keeps
            loss magnitude comparable to simple weighting for logging).
        """
        if self.loss_weighting == 'simple':
            return torch.ones_like(t, dtype=torch.float32)
        # Min-SNR-γ
        alpha_bar = self.diffusion.alphas_cumprod[t].float()
        snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)
        w = torch.clamp(snr, max=self.snr_gamma)
        # Normalize so expected weight ≈ 1 (for stable logging / lr tuning)
        w = w / w.mean().clamp(min=1e-8)
        return w

    def _apply_conv_refine(self, pred_patches):
        """If conv_refine is enabled, apply residual Conv post-processing in
        image space to smooth patch boundaries. Returns refined patches
        (still in patch-space shape for downstream loss computation)."""
        if self.conv_refine is None:
            return pred_patches
        img_size = int(self.patch_size * (self.num_patches ** 0.5))
        pred_img = self.unpatchify(pred_patches, img_size=img_size)
        pred_img = pred_img + self.conv_refine(pred_img)
        return self.patchify(pred_img)

    def _encode_with_indicators(self, mixed_imgs, noisy_mask, t=None):
        """Encode with clean/noisy indicator embeddings + time conditioning.

        Each patch token receives:
          1. Patch pixel embedding (from Conv2d patch_embed)
          2. Clean/noisy indicator embedding (tells encoder which patches are anchors)
          3. Positional embedding (added inside encoder's forward_patches)
          4. Time conditioning:
             - DiTEncoder: per-block adaLN-Zero with time embedding c
             - ViTEncoder: additive time_emb broadcast to all patches (fallback)

        Args:
            mixed_imgs: (B, C, H, W) image with clean + noisy patches mixed
            noisy_mask: (B, N) bool mask, True = noisy, False = clean
            t: (B,) integer timesteps
        Returns:
            cls_token: (B, D), patch_tokens: (B, N, D)
        """
        enc = self.encoder
        B = mixed_imgs.shape[0]
        x = enc.patch_embed(mixed_imgs)  # (B, N, D)

        # Optional: add indicator embeddings per patch
        if self.use_indicators:
            indicator = torch.where(
                noisy_mask.unsqueeze(-1),
                self.noise_indicator.expand(B, x.shape[1], -1),
                self.clean_indicator.expand(B, x.shape[1], -1),
            )
            x = x + indicator

        if self._use_dit_encoder:
            # DiTEncoder: time enters through per-block adaLN-Zero
            c = self.time_embed(t)  # (B, D)
            return enc.forward_patches(x, c)
        else:
            # ViTEncoder fallback: additive time embedding once before blocks
            if t is not None and self.time_embed is not None:
                t_emb = self.time_embed(t)
                x = x + t_emb.unsqueeze(1)
            cls = enc.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + enc.pos_embed
            for blk in enc.blocks:
                x = blk(x)
            x = enc.norm(x)
            return x[:, 0], x[:, 1:]

    def forward(self, imgs, epoch=0):
        """
        Args:
            imgs: (B, C, H, W) clean images
            epoch: current epoch for curriculum scheduling

        Returns:
            loss: reconstruction loss
            log_dict: dict with metrics for logging
        """
        if self.flow_matching and self.naive_ddpm:
            if self.rf_mae_enabled:
                return self._forward_naive_rf_mae(imgs, epoch)
            return self._forward_naive_rf(imgs, epoch)
        if self.naive_ddpm:
            return self._forward_naive_ddpm(imgs, epoch)
        if self.naive_mae:
            return self._forward_naive_mae(imgs, epoch)
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

        # Encode with clean/noisy indicators + time embedding
        cls_token, patch_tokens = self._encode_with_indicators(mixed_imgs, noisy_mask, t)

        # Decode
        pred = self.decoder(patch_tokens)  # (B, N, patch_dim)
        pred = self._apply_conv_refine(pred)

        if self.predict_noise:
            # Noise prediction: target is the noise added to patches
            # For clean patches, noise is zero (already zeroed in apply_patch_noise)
            target = noise
        else:
            # Pixel reconstruction: target is the clean patches
            target = target_patches

        # Primary loss on noisy regions (weighted for ε prediction only)
        w = self._eps_weight(t) if self.predict_noise else None
        noisy_loss = self._masked_mse(pred, target, noisy_mask, weight=w)

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

    def _encode_with_time(self, noisy_imgs, t):
        """Encode with timestep conditioning.

        Uses DiTEncoder's per-block adaLN-Zero modulation: time embedding
        is fed into every transformer block, producing scale/shift/gate for
        both attention and MLP sub-layers. This is the standard conditioning
        mechanism for transformer-based DDPM (DiT, PixArt, SD3).
        """
        time_emb = self.time_embed(t)  # (B, D)
        return self.encoder(noisy_imgs, time_emb)

    def _forward_naive_rf(self, imgs, epoch=0):
        """Naive Rectified Flow on ViT (SD3 / FLUX style, unconditional).

        - t ~ logit-normal (or uniform) in (0, 1)
        - x_t = (1-t)*x_0 + t*ε (linear interpolation)
        - Model predicts v = ε − x_0 (velocity)
        - Loss: MSE(v̂, v) (flow matching loss)
        - NOTE: we feed t × (num_timesteps-1) into the sinusoidal time_embed
          so the scalar passed to the encoder spans the same range as DDPM.
        """
        B = imgs.shape[0]
        device = imgs.device
        img_size = int(self.patch_size * (self.num_patches ** 0.5))

        target_patches = self.patchify(imgs)                       # (B, N, patch_dim)
        t = self.rf.sample_t(B, device)                            # (B,) in (0, 1)
        x_t_patches, v, eps = self.rf.add_noise(target_patches, t)
        x_t_imgs = self.unpatchify(x_t_patches, img_size=img_size)

        # Time embedding: scale t ∈ (0,1) to the integer range used by sinusoidal
        t_int = (t * (self.diffusion.num_timesteps - 1)).long()
        cls_token, patch_tokens = self._encode_with_time(x_t_imgs, t_int)

        pred_v = self.decoder(patch_tokens)                         # (B, N, patch_dim)
        pred_v = self._apply_conv_refine(pred_v)

        # Flow matching loss (simple MSE, no extra weighting needed when using
        # logit-normal t sampling — the sampler already emphasizes informative t).
        loss = F.mse_loss(pred_v, v)

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': loss.detach(),
            'clean_loss': torch.tensor(0.0, device=device),
            't_min': 0,
            't_max': self.diffusion.num_timesteps,
            'clean_ratio': 0.0,
            't_mean': t.mean().detach(),
        }
        return loss, log_dict

    def _forward_naive_rf_mae(self, imgs, epoch=0):
        """RF + MaskDiT-style mask replacement + v-prediction.

        Motivation: naive RF at 224×16 produces impressionistic tiles because
        each patch is predicted largely independently. Replacing a random
        fraction of patch tokens with a learnable mask_token BEFORE the
        encoder forces attention to reconstruct them from context, which
        should add global coherence at the expense of per-patch sharpness.

        Differences from classic MAE:
          - Symmetric: all N tokens go through the DiTEncoder (no asymmetric
            encoder-decoder). The minimal Linear head is per-token anyway.
          - Mask ratio r ~ U(0, rf_mae_max_mask) per step, including r ≈ 0
            so the model still sees the clean-input distribution used at
            inference (no OOD gap during sampling).
          - Loss on all tokens: L_visible (primary) + λ · L_masked (aux).
        """
        B = imgs.shape[0]
        device = imgs.device
        img_size = int(self.patch_size * (self.num_patches ** 0.5))
        N = self.num_patches

        # RF forward
        target_patches = self.patchify(imgs)
        t = self.rf.sample_t(B, device)
        x_t_patches, v, eps = self.rf.add_noise(target_patches, t)
        x_t_imgs = self.unpatchify(x_t_patches, img_size=img_size)

        # Patch-embed (then we'll swap in mask_token on selected positions)
        patch_tokens = self.encoder.patch_embed(x_t_imgs)  # (B, N, D)

        # Per-step mask ratio r ~ U(0, rf_mae_max_mask). Same r for all samples
        # in the batch; masked positions chosen independently per sample.
        r = float(torch.empty((), device=device).uniform_(0.0, self.rf_mae_max_mask))
        len_mask = int(round(N * r))

        if len_mask > 0:
            rand = torch.rand(B, N, device=device)
            ids_shuffle = torch.argsort(rand, dim=1)
            ids_mask = ids_shuffle[:, :len_mask]
            mask = torch.zeros(B, N, device=device)
            mask.scatter_(1, ids_mask, 1.0)        # 1 = masked, 0 = visible
            m = mask.unsqueeze(-1)                  # (B, N, 1)
            mask_token = self.rf_mask_token.expand(B, N, -1)
            patch_tokens = patch_tokens * (1 - m) + mask_token * m
        else:
            mask = torch.zeros(B, N, device=device)

        # Time conditioning through DiTEncoder (forward_patches applies
        # cls_token + pos_embed + adaLN-Zero blocks)
        t_int = (t * (self.diffusion.num_timesteps - 1)).long()
        c = self.time_embed(t_int)
        cls_token, enc_patch_tokens = self.encoder.forward_patches(patch_tokens, c)

        pred_v = self.decoder(enc_patch_tokens)
        pred_v = self._apply_conv_refine(pred_v)

        visible_mask = (mask == 0)
        masked_mask = (mask == 1)
        vis_loss = self._masked_mse(pred_v, v, visible_mask)
        if masked_mask.sum() > 0:
            mask_loss = self._masked_mse(pred_v, v, masked_mask)
        else:
            mask_loss = torch.tensor(0.0, device=device)

        loss = vis_loss + self.mae_aux_weight * mask_loss

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': vis_loss.detach(),
            'clean_loss': mask_loss.detach(),
            't_min': 0,
            't_max': self.diffusion.num_timesteps,
            'clean_ratio': 1.0 - r,              # fraction of visible patches
            't_mean': t.mean().detach(),
        }
        return loss, log_dict

    def _forward_naive_ddpm(self, imgs, epoch=0):
        """Naive DDPM-ViT pretraining (standard DDPM training on a ViT).

        - Per-image t ~ Uniform[0, T) (standard DDPM).
        - Add noise to the ENTIRE image (no patch mixing, no clean anchors).
        - ViT encoder receives the noisy image with timestep conditioning
          injected via the cls_token.
        - Decoder predicts noise epsilon for every patch.
        - Loss: MSE on all patches (standard DDPM training objective).
        """
        B = imgs.shape[0]
        device = imgs.device
        img_size = int(self.patch_size * (self.num_patches ** 0.5))

        # Standard DDPM forward: uniform t, noise the whole image
        target_patches = self.patchify(imgs)  # (B, N, patch_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        noisy_patches, noise = self.diffusion.add_noise(target_patches, t)
        noisy_imgs = self.unpatchify(noisy_patches, img_size=img_size)

        # Encoder with time conditioning
        cls_token, patch_tokens = self._encode_with_time(noisy_imgs, t)

        # Decoder predicts noise on all patches
        pred_noise = self.decoder(patch_tokens)  # (B, N, patch_dim)
        pred_noise = self._apply_conv_refine(pred_noise)

        # Weighted MSE (uniform or Min-SNR-γ)
        w = self._eps_weight(t)                                  # (B,)
        per_sample = ((pred_noise - noise) ** 2).mean(dim=(1, 2))  # (B,)
        loss = (w * per_sample).mean()

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': loss.detach(),
            'clean_loss': torch.tensor(0.0, device=device),
            't_min': 0,
            't_max': self.diffusion.num_timesteps,
            'clean_ratio': 0.0,
            't_mean': t.float().mean().detach(),
        }
        return loss, log_dict

    def _forward_naive_mae(self, imgs, epoch=0):
        """Naive MAE pretraining (Kaiming He et al. 2021).

        - No diffusion noise — pure random masking.
        - mask_ratio of patches are dropped (typically 0.75).
        - Encoder only processes (1 - mask_ratio) clean visible patches (efficient).
        - Decoder fills mask tokens at dropped positions, reconstructs pixels.
        - Loss: MSE on MASKED patches only (reconstruct what encoder didn't see).
        """
        B = imgs.shape[0]
        target_patches = self.patchify(imgs)  # (B, N, patch_dim)

        # Encoder sees clean visible patches (mask_ratio fraction masked out)
        cls_token, visible_tokens, ids_restore, mask = self.encoder.forward_masked(
            imgs, mask_ratio=self.mask_ratio
        )
        # mask: (B, N) with 1 = masked, 0 = visible

        # Decoder reconstructs all patches (but only masked are scored)
        pred = self.decoder.forward_masked(visible_tokens, ids_restore)

        visible_mask = (mask == 0)
        masked_mask = (mask == 1)

        # MAE loss: reconstruction MSE on masked patches only
        loss = self._masked_mse(pred, target_patches, masked_mask)

        # Track visible loss as diagnostic (not in total loss)
        vis_loss = self._masked_mse(pred, target_patches, visible_mask)

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': loss.detach(),       # report masked loss as 'noisy_loss' for logger compat
            'clean_loss': vis_loss.detach(),   # report visible loss as 'clean_loss'
            't_min': 0, 't_max': 0,            # no timesteps in naive MAE
            'clean_ratio': 1 - self.mask_ratio,
            't_mean': torch.tensor(0.0, device=imgs.device),
        }
        return loss, log_dict

    def _forward_dual(self, imgs, epoch=0):
        """Dual-decoder pretraining: shared encoder + two heads.

        Design:
          - Encoder sees clean(clean_ratio) + noisy(1 - clean_ratio) patches
            (no MAE masking — clean patches serve as spatial anchors)
          - t sampled from curriculum-controlled [t_min, t_max] range
            (if curriculum is degenerate to [0, T], this is standard DDPM uniform)
          - decoder      → predicts noise ε for each patch (DDPM target)
          - decoder_pix  → reconstructs clean pixel values (MAE target)
          - Loss on the NOISY patches only (clean patches are trivial for both heads)
          - Combined: L = L_eps + pixel_loss_weight * L_pix
        """
        B = imgs.shape[0]
        device = imgs.device

        # Use curriculum-controlled t range (allows ablation: curriculum vs uniform)
        curriculum_state = self.curriculum.get_state(epoch)
        t_min, t_max = curriculum_state['t_min'], curriculum_state['t_max']

        target_patches = self.patchify(imgs)  # (B, N, patch_dim)

        t = self.diffusion.sample_timesteps(B, t_min, t_max, device)
        # clean_ratio fraction stays clean (spatial anchors), rest is noisy
        noisy_mask = self.diffusion.generate_noisy_mask(
            B, self.num_patches, self.clean_ratio, device
        )
        mixed_patches, noise, noisy_mask = self.diffusion.apply_patch_noise(
            target_patches, noisy_mask, t
        )

        img_size = int(self.patch_size * (self.num_patches ** 0.5))
        mixed_imgs = self.unpatchify(mixed_patches, img_size=img_size)

        # Shared encoder with clean/noisy indicators + time
        cls_token, patch_tokens = self._encode_with_indicators(mixed_imgs, noisy_mask, t)

        # Two decoders predict different targets
        pred_eps = self.decoder(patch_tokens)         # noise prediction
        pred_pix = self.decoder_pix(patch_tokens)     # pixel reconstruction
        pred_eps = self._apply_conv_refine(pred_eps)
        pred_pix = self._apply_conv_refine(pred_pix)

        # Loss on the 75% noisy patches. Weight ε loss by Min-SNR-γ if enabled;
        # pix loss stays unweighted (target is pixel values, not noise).
        w = self._eps_weight(t)
        loss_eps = self._masked_mse(pred_eps, noise, noisy_mask, weight=w)
        loss_pix = self._masked_mse(pred_pix, target_patches, noisy_mask)

        loss = loss_eps + self.pixel_loss_weight * loss_pix

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': loss_eps.detach(),   # eps loss reported as "noisy_loss" for logger compat
            'clean_loss': loss_pix.detach(),   # pixel loss reported as "clean_loss"
            't_min': t_min,
            't_max': t_max,
            'clean_ratio': self.clean_ratio,
            't_mean': t.float().mean().detach(),
        }
        return loss, log_dict

    def _masked_mse(self, pred, target, mask, weight=None):
        """Compute MSE loss only on masked (selected) patches.

        Args:
            pred, target: (B, N, patch_dim)
            mask: (B, N) bool selection mask
            weight: (B,) optional per-sample scalar weight (for loss weighting
                schemes like Min-SNR-γ). If None, uniform weighting.
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = (pred - target) ** 2
        diff = diff.mean(dim=-1)  # (B, N)
        mask_f = mask.float()
        if weight is not None:
            # Per-sample mean of masked-patch losses, then weighted average
            denom = mask_f.sum(dim=1).clamp(min=1e-8)     # (B,)
            per_sample = (diff * mask_f).sum(dim=1) / denom  # (B,)
            return (weight * per_sample).mean()
        return (diff * mask_f).sum() / mask_f.sum()

    def get_encoder(self):
        """Return the encoder for downstream evaluation."""
        return self.encoder
