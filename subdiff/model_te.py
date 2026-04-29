"""TransformerEngine-based classical MAE for Run 12e Stage 1.

Asymmetric encoder-decoder design (He et al. 2021):
  - Encoder sees ONLY visible 25% patches (4× faster than full-N)
  - Decoder sees full N (visible encoded + learnable mask tokens)
  - Decoder is shallower / narrower than encoder
  - Loss = MSE on masked patches only

Differences from canonical MAE for our μ-generator role:
  - We keep mask_ratio = 0.75 (canonical), 25% visible serves as the
    "prompt" at inference (compatible with Run X / Run 12 cold-RF).
  - Encoder + decoder both built from te.TransformerLayer for fp8 GEMMs
    on Hopper (te.fp8_autocast wraps forward).
  - No t-conditioning: pure MAE (no diffusion, no v-head).
"""

import torch
import torch.nn as nn
import transformer_engine.pytorch as te


def _sincos_pos_embed_2d(num_patches, dim, device, dtype):
    grid_size = int(num_patches ** 0.5)
    assert grid_size * grid_size == num_patches
    grid_h = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid_w = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    assert dim % 4 == 0
    half = dim // 4
    omega = torch.arange(half, device=device, dtype=torch.float32) / half
    omega = 1.0 / (10000 ** omega)
    out = []
    for i in range(2):
        pos = grid[i].flatten()
        emb_sin = torch.sin(pos[:, None] * omega[None, :])
        emb_cos = torch.cos(pos[:, None] * omega[None, :])
        out.append(torch.cat([emb_sin, emb_cos], dim=-1))
    pos_embed = torch.cat(out, dim=-1)
    return pos_embed.to(dtype)


class MAEClassicViT(nn.Module):
    """Classical asymmetric MAE: encoder-only-on-visible, transformer decoder.

    Args:
        img_size, patch_size, in_chans: standard ViT inputs
        embed_dim, depth, num_heads:    encoder spec (default ViT-B)
        decoder_dim, decoder_depth, decoder_num_heads: decoder spec
                                        (smaller per MAE paper)
        clean_ratio:    fraction of visible patches at training/inference.
                        canonical MAE uses 0.25 (= mask_ratio 0.75).
        params_dtype:   bf16 to match TE fp8 path
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        clean_ratio=0.25,
        params_dtype=torch.bfloat16,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.clean_ratio = clean_ratio

        # ---- Encoder side ----
        # Patch embed (Conv2d, kept bf16). Operates on the FULL image; we
        # later gather only visible patch embeddings for the encoder.
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.register_buffer(
            'enc_pos_embed_patches',
            _sincos_pos_embed_2d(self.num_patches, embed_dim,
                                 device='cpu', dtype=torch.float32),
        )
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

        self.encoder_blocks = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=embed_dim,
                ffn_hidden_size=4 * embed_dim,
                num_attention_heads=num_heads,
                self_attn_mask_type='no_mask',
                hidden_dropout=0.0,
                attention_dropout=0.0,
                params_dtype=params_dtype,
            )
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ---- Decoder side ----
        # Project encoder output to (smaller) decoder dim
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_dim)
        # Learnable mask token shared across all masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Decoder has its own (smaller) sin-cos pos embed
        self.register_buffer(
            'dec_pos_embed_patches',
            _sincos_pos_embed_2d(self.num_patches, decoder_dim,
                                 device='cpu', dtype=torch.float32),
        )
        self.dec_cls_pos_embed = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.dec_cls_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=decoder_dim,
                ffn_hidden_size=4 * decoder_dim,
                num_attention_heads=decoder_num_heads,
                self_attn_mask_type='no_mask',
                hidden_dropout=0.0,
                attention_dropout=0.0,
                params_dtype=params_dtype,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # Pixel prediction head
        self.decoder_pred = nn.Linear(decoder_dim, in_chans * (patch_size ** 2))

    # ------------------------------------------------------------------
    # Patchify / unpatchify
    # ------------------------------------------------------------------
    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        p = self.patch_size
        x = imgs.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return x

    def unpatchify(self, patches, img_size=None):
        if img_size is None:
            img_size = self.img_size
        B, N, _ = patches.shape
        p = self.patch_size
        H = W = img_size
        x = patches.reshape(B, H // p, W // p, self.in_chans, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, self.in_chans, H, W)
        return x

    # ------------------------------------------------------------------
    # Random shuffling: returns visible token indices and restore indices.
    # Standard MAE algorithm.
    # ------------------------------------------------------------------
    @staticmethod
    def _random_masking(x, mask_ratio):
        """x: (B, N, D) or (B, N) — anything indexable on dim 1.
        Returns:
            visible: (B, len_keep, D)
            mask:    (B, N) — 1 = masked, 0 = visible
            ids_restore: (B, N) — argsort to undo shuffle
        """
        B, N = x.shape[0], x.shape[1]
        len_keep = int(round(N * (1 - mask_ratio)))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)             # (B, N)
        ids_restore = ids_shuffle.argsort(dim=1)       # (B, N)
        ids_keep = ids_shuffle[:, :len_keep]           # (B, len_keep)
        # Gather visible tokens
        if x.dim() == 3:
            D = x.shape[2]
            visible = torch.gather(
                x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        else:
            visible = torch.gather(x, 1, ids_keep)
        # Build mask: 1 where masked, 0 where visible
        mask = torch.ones(B, N, device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return visible, mask, ids_restore

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, imgs):
        B = imgs.shape[0]
        device = imgs.device

        # ---- Patch embed (full image) ----
        x_full = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (B, N, D)
        # Add patch positional embedding (no CLS yet)
        enc_pos = self.enc_pos_embed_patches.to(device=device, dtype=x_full.dtype)
        x_full = x_full + enc_pos.unsqueeze(0)

        # ---- Random shuffle, keep visible 25% ----
        visible, mask, ids_restore = self._random_masking(
            x_full, mask_ratio=1 - self.clean_ratio)
        # mask: (B, N), 1 = masked (loss target), 0 = visible

        # ---- Prepend CLS token (with its learnable pos) ----
        cls = (self.cls_token + self.cls_pos_embed).expand(B, -1, -1).to(visible.dtype)
        enc_in = torch.cat([cls, visible], dim=1)  # (B, 1+len_keep, D)

        # ---- Encoder (only sees visible 25% + CLS — the speedup) ----
        x = enc_in.transpose(0, 1).contiguous()  # (S, B, D)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = x.transpose(0, 1).contiguous()       # (B, 1+len_keep, D)
        x = self.encoder_norm(x)

        # ---- Project to decoder dim ----
        x = self.encoder_to_decoder(x)           # (B, 1+len_keep, D_dec)

        # ---- Re-insert mask tokens at masked positions, unshuffle ----
        cls_dec = x[:, :1, :]                    # (B, 1, D_dec)
        x_visible = x[:, 1:, :]                  # (B, len_keep, D_dec)
        D_dec = self.decoder_dim
        # Pad with mask tokens up to N, then reorder via ids_restore
        N = self.num_patches
        len_keep = x_visible.shape[1]
        mask_pad = self.mask_token.expand(B, N - len_keep, -1).to(x.dtype)
        x_full_dec = torch.cat([x_visible, mask_pad], dim=1)  # (B, N, D_dec)
        x_full_dec = torch.gather(
            x_full_dec, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D_dec))
        # Add decoder pos embed
        dec_pos = self.dec_pos_embed_patches.to(device=device, dtype=x.dtype)
        x_full_dec = x_full_dec + dec_pos.unsqueeze(0)
        # Prepend CLS (with its own decoder cls pos)
        cls_dec = cls_dec + self.dec_cls_pos_embed.to(x.dtype)
        x_dec = torch.cat([cls_dec, x_full_dec], dim=1)  # (B, 1+N, D_dec)

        # ---- Decoder (full N + CLS; cheaper than encoder due to smaller dim) ----
        x = x_dec.transpose(0, 1).contiguous()
        for blk in self.decoder_blocks:
            x = blk(x)
        x = x.transpose(0, 1).contiguous()
        x = self.decoder_norm(x)

        # ---- Predict patches (drop CLS) ----
        pred = self.decoder_pred(x[:, 1:])  # (B, N, in_chans*p²)

        # ---- Loss: MSE on masked patches only ----
        target = self.patchify(imgs).to(pred.dtype)
        diff_sq = (pred.float() - target.float()) ** 2  # (B, N, patch_dim)
        per_patch = diff_sq.mean(dim=-1)                # (B, N)
        masked_f = mask.float()
        denom = masked_f.sum().clamp_min(1.0)
        loss = (per_patch * masked_f).sum() / denom

        visible_f = 1.0 - masked_f
        denom_v = visible_f.sum().clamp_min(1.0)
        visible_loss = (per_patch * visible_f).sum() / denom_v

        log_dict = {
            'loss': loss.detach(),
            'noisy_loss': loss.detach(),
            'clean_loss': visible_loss.detach(),
            't_min': 0,
            't_max': 0,
            'clean_ratio': self.clean_ratio,
            't_mean': torch.tensor(0.0, device=device),
        }
        return loss, log_dict

    # ------------------------------------------------------------------
    # Inpaint inference: μ-generator API for Stage 2 cold-RF Refiner.
    # Takes a fixed noisy_mask (which patches are prompt vs predict-target)
    # rather than random, and returns predicted pixel patches at all positions.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_mu(self, imgs, noisy_mask):
        """
        Args:
            imgs:       (B, C, H, W) — clean image (only the visible patches
                        from this image are actually used)
            noisy_mask: (B, N) bool — True at positions to predict (target),
                        False at prompt positions
        Returns:
            pred:       (B, N, in_chans*p²) — predicted patches at all positions
                        (caller can use .compute_mu output as μ for cold-RF chain)
        """
        B = imgs.shape[0]
        device = imgs.device

        # Patch embed full image
        x_full = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        enc_pos = self.enc_pos_embed_patches.to(device=device, dtype=x_full.dtype)
        x_full = x_full + enc_pos.unsqueeze(0)

        # Build "ids_restore" matching noisy_mask:
        #   visible positions go first in encoder, masked second; we then
        #   unshuffle via ids_restore to put them back in original positions.
        # noisy_mask: True = mask (predict). visible_idx = where mask is False.
        N = self.num_patches
        # Per-batch sorting that places visible (False) before masked (True)
        # — stable sort keeps relative order within each group.
        order = torch.argsort(noisy_mask.int(), dim=1, stable=True)  # (B, N)
        ids_restore = torch.argsort(order, dim=1, stable=True)
        len_keep = (noisy_mask == False).sum(dim=1)[0].item()
        # Sanity: in our cold-RF inference we use a uniform clean_ratio per
        # batch row (same number of visible per row). If asymmetric, this
        # path needs per-row gather.
        # Gather visible
        D = x_full.shape[-1]
        visible = torch.gather(
            x_full, 1, order[:, :len_keep].unsqueeze(-1).expand(-1, -1, D))

        cls = (self.cls_token + self.cls_pos_embed).expand(B, -1, -1).to(visible.dtype)
        enc_in = torch.cat([cls, visible], dim=1)
        x = enc_in.transpose(0, 1).contiguous()
        for blk in self.encoder_blocks:
            x = blk(x)
        x = x.transpose(0, 1).contiguous()
        x = self.encoder_norm(x)
        x = self.encoder_to_decoder(x)

        cls_dec = x[:, :1, :]
        x_visible = x[:, 1:, :]
        D_dec = self.decoder_dim
        mask_pad = self.mask_token.expand(B, N - len_keep, -1).to(x.dtype)
        x_full_dec = torch.cat([x_visible, mask_pad], dim=1)
        x_full_dec = torch.gather(
            x_full_dec, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D_dec))
        dec_pos = self.dec_pos_embed_patches.to(device=device, dtype=x.dtype)
        x_full_dec = x_full_dec + dec_pos.unsqueeze(0)
        cls_dec = cls_dec + self.dec_cls_pos_embed.to(x.dtype)
        x_dec = torch.cat([cls_dec, x_full_dec], dim=1)

        x = x_dec.transpose(0, 1).contiguous()
        for blk in self.decoder_blocks:
            x = blk(x)
        x = x.transpose(0, 1).contiguous()
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x[:, 1:])
        return pred
