"""
ViT encoder and lightweight decoder for SubDiff pretraining.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


def build_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    """Fixed 2D sin-cos positional embedding (DiT / MAE style).

    Splits embed_dim into 4 quarters: sin/cos for row, sin/cos for column.
    Adjacent positions have similar embeddings, breaking permutation
    invariance even at initialization (no training required).

    Args:
        embed_dim: feature dimension (must be divisible by 4)
        grid_size: int (side length of the patch grid)
        cls_token: if True, prepend a zero vector for cls position
    Returns:
        (1, grid_size*grid_size [+1], embed_dim) tensor
    """
    assert embed_dim % 4 == 0, f"embed_dim must be divisible by 4, got {embed_dim}"
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size); (W, H) order
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    # Half the channels encode row position, the other half encode column
    half = embed_dim // 2
    emb_h = _sincos_1d(half, grid[0].reshape(-1))   # W component (width index)
    emb_w = _sincos_1d(half, grid[1].reshape(-1))   # H component (height index)
    # Stack: (N, embed_dim) where N = grid_size**2
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)  # (1, N, D)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, 1, embed_dim), pos_embed], dim=1)
    return pos_embed


def _sincos_1d(embed_dim, positions):
    """1D sin-cos embedding: half dims for sin, half for cos."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = np.einsum('m,d->md', positions.reshape(-1), omega)  # (N, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (N, D)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = attn.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_norm=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def modulate(x, scale, shift):
    """x * (1 + scale) + shift, broadcast over sequence dim."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero time conditioning (DiT-style).

    Each block receives a time embedding `c` and produces per-instance
    (scale, shift, gate) modulation coefficients for the attention and MLP
    sub-layers. adaLN parameters are zero-initialized so each block starts
    as an identity, making training much more stable in early epochs.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_norm=False):
        super().__init__()
        # Use elementwise_affine=False because adaLN provides its own scale/shift
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio)
        # adaLN-Zero modulation: MLP(t) -> 6 * dim (scale1/shift1/gate1/scale2/shift2/gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def _init_adaln_zero(self):
        # Zero-init the last linear so that each block starts as identity.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        """
        Args:
            x: (B, N, D) tokens
            c: (B, D) conditioning (time embedding, etc.)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), scale_msa, shift_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x


class DiTEncoder(nn.Module):
    """ViT encoder with DiT-style adaLN-Zero time conditioning.

    Same patch_embed / positional embed / cls_token structure as ViTEncoder,
    but every transformer block is a DiTBlock that consumes a time embedding.
    Timestep conditioning is applied per-block (not just at the cls_token),
    which empirically stabilizes DDPM training on transformer backbones.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, qk_norm=False,
                 pos_embed_type='sincos'):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = img_size // patch_size
        self.pos_embed_type = pos_embed_type

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if pos_embed_type == 'sincos':
            # Fixed 2D sin-cos (DiT standard). Ensures ViT is NOT permutation
            # invariant even at init; no training required to learn spatial prior.
            pos_embed = build_2d_sincos_pos_embed(embed_dim, self.grid_size, cls_token=True)
            self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        else:
            # Learnable (legacy). Can collapse if training signal is weak.
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, qk_norm=qk_norm) for _ in range(depth)
        ])
        # Final norm also adaLN-conditioned (DiT's final_layer adaLN)
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        if self.pos_embed_type == 'learnable':
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)
        # Zero-init all adaLN modulation weights (DiT paper's "adaLN-Zero")
        for blk in self.blocks:
            blk._init_adaln_zero()
        nn.init.zeros_(self.adaLN_final[-1].weight)
        nn.init.zeros_(self.adaLN_final[-1].bias)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_patches(self, patch_tokens, c):
        """Forward starting from precomputed patch token embeddings.

        Useful when the caller wants to inject additional signals (e.g., clean/
        noisy indicators from SubDiff) into the patch tokens before the
        transformer stack.

        Args:
            patch_tokens: (B, N, D) — already patch_embedded, NOT yet prepended
                with cls_token and NOT yet positional-embedded
            c: (B, D) time embedding

        Returns:
            cls_token: (B, D), patch_tokens: (B, N, D)
        """
        B = patch_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patch_tokens], dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x, c)

        shift, scale = self.adaLN_final(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), scale, shift)
        return x[:, 0], x[:, 1:]

    def forward(self, x, c):
        """
        Args:
            x: (B, C, H, W) input image
            c: (B, D) time embedding
        Returns:
            cls_token: (B, D)
            patch_tokens: (B, N, D)
        """
        patch_tokens = self.patch_embed(x)
        return self.forward_patches(patch_tokens, c)


class ViTEncoder(nn.Module):
    """ViT encoder for SubDiff pretraining."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, qk_norm=False,
                 pos_embed_type='sincos'):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = img_size // patch_size
        self.pos_embed_type = pos_embed_type

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if pos_embed_type == 'sincos':
            pos_embed = build_2d_sincos_pos_embed(embed_dim, self.grid_size, cls_token=True)
            self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, qk_norm=qk_norm) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        if self.pos_embed_type == 'learnable':
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) image with noisy + clean patches
        Returns:
            cls_token: (B, D) for linear probe
            patch_tokens: (B, N, D) for reconstruction
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1:]  # cls_token, patch_tokens

    def forward_masked(self, x, mask_ratio=0.25):
        """MAE-style forward: encoder only sees (1 - mask_ratio) visible patches.

        Args:
            x: (B, C, H, W) input image (already noised if used for diffusion pretraining)
            mask_ratio: fraction of patches to mask out (encoder won't see them)

        Returns:
            cls_token: (B, D)
            visible_tokens: (B, N_vis, D) encoded visible patches
            ids_restore: (B, N) indices that restore the original order
            mask: (B, N) binary mask (1 = masked, 0 = visible)
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)

        # Add pos embed (skip cls pos embed at index 0)
        x = x + self.pos_embed[:, 1:, :]

        # Random masking
        N = x.shape[1]
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        # Build mask: 0 = visible, 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Prepend cls token (with its own pos embed)
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x_vis = torch.cat([cls, x_vis], dim=1)

        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)

        return x_vis[:, 0], x_vis[:, 1:], ids_restore, mask


class Decoder(nn.Module):
    """Lightweight decoder for pixel reconstruction."""

    def __init__(self, patch_size=16, num_patches=196,
                 encoder_dim=768, decoder_dim=512, depth=4, num_heads=8,
                 qk_norm=False):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, qk_norm=qk_norm) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size ** 2 * 3)
        self.patch_size = patch_size

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (B, N, encoder_dim)
        Returns:
            pred: (B, N, patch_size**2 * 3) predicted pixel values per patch
        """
        x = self.proj(patch_tokens)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x)

    def forward_masked(self, visible_tokens, ids_restore):
        """MAE-style decoder: fill masked positions with mask_token, unshuffle, decode.

        Args:
            visible_tokens: (B, N_vis, encoder_dim) encoded visible patches
            ids_restore: (B, N) indices to restore original patch order

        Returns:
            pred: (B, N, patch_dim) predicted target (noise or pixel) for all patches
        """
        B, N_vis, _ = visible_tokens.shape
        N = ids_restore.shape[1]

        x = self.proj(visible_tokens)  # (B, N_vis, decoder_dim)
        # Append mask tokens for the masked positions
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)
        x = torch.cat([x, mask_tokens], dim=1)  # (B, N, decoder_dim)
        # Unshuffle back to original order
        x = torch.gather(x, dim=1,
                         index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x)
