"""fp8 + p4 + batch=64 memory feasibility (Phase 0 of Run 12e).

Builds a minimal ViT-B at p4 (3136 tokens) using TransformerEngine
TransformerLayer, runs 5 fwd+bwd steps under fp8_autocast, reports peak
GPU memory + step time.

If peak < 60 GB on H100 80GB → Phase 1 (cold_rf cross-resolution adapter)
becomes worthwhile. If still OOM, fall back to gradient checkpointing or
smaller batch.

Compares against: bf16+p4 batch=64 was 80+ GB OOM (probe 530069). Bench
goal: fp8 should bring it to ~40 GB.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

sys.path.insert(0, '/lustre1/work/c30636/test/subdiff')
from scripts.pretrain import _enable_hpc_speedups, _select_sdpa_backend


class FP8ViTB(nn.Module):
    """ViT-B encoder with TE TransformerLayer blocks. Same shape budget as
    Run 12 refiner: 12 layers, 768 dim, 12 heads, FFN 4×."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, out_dim=None):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embed (kept bf16; fp8 only inside transformer blocks)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=embed_dim,
                ffn_hidden_size=4 * embed_dim,
                num_attention_heads=num_heads,
                self_attn_mask_type='no_mask',
                hidden_dropout=0.0,
                attention_dropout=0.0,
                params_dtype=torch.bfloat16,
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        if out_dim is None:
            out_dim = in_chans * (patch_size ** 2)  # v-pred / x_0 patches
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, imgs):
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        # TE TransformerLayer is seq-first by default
        x = x.transpose(0, 1).contiguous()  # (S, B, D)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(0, 1).contiguous()  # (B, S, D)
        x = self.norm(x)
        out = self.head(x[:, 1:])  # exclude CLS, predict per-patch
        return out


def main():
    _enable_hpc_speedups()
    _select_sdpa_backend('cudnn')

    device = torch.device('cuda')
    torch.manual_seed(0)

    BS = 64
    IMG = 224
    PATCH = 4

    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)')
    print(f'TE version: {te.__version__ if hasattr(te, "__version__") else "?"}')

    model = FP8ViTB(img_size=IMG, patch_size=PATCH, embed_dim=768,
                    depth=12, num_heads=12).to(device).bfloat16()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {n_params:.1f}M params, p={PATCH}, '
          f'N={(IMG//PATCH)**2 + 1} (incl. CLS)')

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    imgs = torch.randn(BS, 3, IMG, IMG, device=device, dtype=torch.bfloat16)
    target_dim = 3 * (PATCH ** 2)
    target = torch.randn(BS, (IMG // PATCH) ** 2, target_dim,
                         device=device, dtype=torch.bfloat16)

    fp8_recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID)

    torch.cuda.reset_peak_memory_stats()

    print(f'\n=== fp8 + p4 + batch={BS} ===')
    for step in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = model(imgs)
            loss = (out.float() - target.float()).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f'  step {step}: loss={loss.item():.5f}  dt={dt*1000:.0f}ms  '
              f'peak={peak_gb:.2f}GB')

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'\n=== verdict ===')
    print(f'  peak alloc: {peak_gb:.2f} GB / {total_gb:.0f} GB '
          f'({peak_gb/total_gb*100:.0f}%)')
    headroom = total_gb - peak_gb
    print(f'  headroom: {headroom:.1f} GB')
    if peak_gb < total_gb * 0.7:
        print(f'  → fp8 + p4 + batch=64 EASILY fits, plenty of room for '
              f'cold_rf mu_model (~6 GB) + DDP overhead')
    elif peak_gb < total_gb * 0.85:
        print(f'  → fits with reasonable margin, cold_rf path viable')
    elif peak_gb < total_gb:
        print(f'  → fits but tight, may need mu_model on separate GPU')
    else:
        print(f'  → OOM (unexpected)')


if __name__ == '__main__':
    main()
