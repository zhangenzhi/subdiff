"""Single-GPU memory feasibility probe for patch_size=4 at batch=64.

Builds a p4 SubDiff (cold_rf disabled, just to keep it simple), runs 5
fwd+bwd steps with bf16 autocast, reports peak GPU memory.
"""

import sys
import time
import yaml

import torch

sys.path.insert(0, '/lustre1/work/c30636/test/subdiff')
from subdiff.model import SubDiff
from scripts.pretrain import _enable_hpc_speedups, _select_sdpa_backend


def main():
    _enable_hpc_speedups()
    _select_sdpa_backend('cudnn')  # FA3 on H100

    device = torch.device('cuda')
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    BS = 64
    IMG = 224
    PATCH = 4

    # Build a minimal p4 SubDiff. Use dual-RF style (predict both x_0 and v)
    # so the model is the same shape we'd actually train. cold_rf=False keeps
    # the path simple — we just want to know if forward+backward fits.
    model = SubDiff(
        img_size=IMG,
        patch_size=PATCH,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        schedule_type='linear',
        total_epochs=1,
        curriculum_cfg={
            't_min_start': 0, 't_min_end': 0,
            't_max_start': 0, 't_max_end': 0,
            'clean_ratio_start': 0.25, 'clean_ratio_end': 0.25,
            'warmup_epochs': 0, 'schedule': 'linear',
        },
        predict_noise=False,
        dual_decoder=True,
        clean_ratio=0.25,
        pixel_loss_weight=1.0,
        qk_norm=True,
        dit_minimal_head=True,
        use_indicators=True,
        pos_embed_type='sincos',
        flow_matching=True,
        rf_t_sampling='logit_normal',
        rf_logit_mean=0.0,
        rf_logit_std=1.0,
        cold_rf=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_patches = (IMG // PATCH) ** 2
    print(f'Model: {n_params:.1f}M params, p={PATCH}, N={n_patches} tokens '
          f'(vs p=8 N={(IMG//8)**2})')
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB total)')

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    imgs = torch.randn(BS, 3, IMG, IMG, device=device)

    print(f'\nRunning 5 fwd+bwd steps at batch={BS}...')
    for step in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, log = model(imgs, epoch=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f'  step {step}: loss={loss.item():.5f}  dt={dt*1000:.0f}ms  '
              f'peak={peak_gb:.2f}GB  reserved={reserved_gb:.2f}GB')

    print('\n=== verdict ===')
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  peak alloc:    {peak_gb:.2f} GB / {total_gb:.0f} GB '
          f'({peak_gb/total_gb*100:.0f}%)')
    print(f'  reserved:      {reserved_gb:.2f} GB / {total_gb:.0f} GB '
          f'({reserved_gb/total_gb*100:.0f}%)')
    headroom = total_gb - reserved_gb
    print(f'  free headroom: {headroom:.1f} GB')
    if peak_gb < total_gb * 0.85:
        print(f'  → batch=64 at p4 FITS (uses {peak_gb/total_gb*100:.0f}% of '
              f'memory, leaves {headroom:.0f} GB margin)')
    elif peak_gb < total_gb:
        print(f'  → batch=64 at p4 fits but tight ({peak_gb/total_gb*100:.0f}%)')
    else:
        print(f'  → OOM')


if __name__ == '__main__':
    main()
