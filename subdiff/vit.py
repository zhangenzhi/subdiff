"""
ViT encoder and lightweight decoder for SubDiff pretraining.
"""

import torch
import torch.nn as nn
from einops import rearrange


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
    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
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
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT encoder for SubDiff pretraining."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
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
                 encoder_dim=768, decoder_dim=512, depth=4, num_heads=8):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.blocks = nn.ModuleList([
            Block(decoder_dim, num_heads) for _ in range(depth)
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
