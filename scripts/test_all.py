"""
Test script: verify data loading, model construction, and forward/backward pass.
"""

import os
import sys
import time
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff
from subdiff.diffusion import PatchDiffusion
from subdiff.curriculum import CurriculumScheduler
from subdiff.data import build_pretrain_dataloader, build_eval_dataloader


def test_data_loading(cfg):
    print("=" * 60)
    print("[1/5] Testing data loading (train)")
    print("=" * 60)
    loader, _ = build_pretrain_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=4,
        num_workers=2,
        distributed=False,
    )
    print(f"  Train dataset size: {len(loader.dataset)}")
    print(f"  Train loader batches: {len(loader)}")

    imgs, labels = next(iter(loader))
    print(f"  Batch images shape: {imgs.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Image value range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"  Labels sample: {labels.tolist()}")

    print("\n  Testing eval data loading (val)...")
    val_loader, _ = build_eval_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=4,
        num_workers=2,
    )
    print(f"  Val dataset size: {len(val_loader.dataset)}")
    val_imgs, val_labels = next(iter(val_loader))
    print(f"  Val batch shape: {val_imgs.shape}")
    print("  [PASS] Data loading OK\n")
    return imgs


def test_diffusion(imgs):
    print("=" * 60)
    print("[2/5] Testing PatchDiffusion")
    print("=" * 60)
    diffusion = PatchDiffusion(num_timesteps=1000, schedule_type='linear')

    # Simulate patches: (B, N, patch_dim) for 16x16 patches
    B = imgs.shape[0]
    patch_size = 16
    num_patches = (224 // patch_size) ** 2
    patch_dim = patch_size ** 2 * 3
    from einops import rearrange
    patches = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    print(f"  Patches shape: {patches.shape}")

    # Test add_noise
    t = diffusion.sample_timesteps(B, 100, 900, patches.device)
    noisy, noise = diffusion.add_noise(patches, t)
    print(f"  Noisy patches shape: {noisy.shape}, noise shape: {noise.shape}")
    print(f"  Timesteps: {t.tolist()}")

    # Test generate_noisy_mask
    mask = diffusion.generate_noisy_mask(B, num_patches, clean_ratio=0.25, device=patches.device)
    print(f"  Noisy mask shape: {mask.shape}, noisy count per sample: {mask.sum(dim=1).tolist()}")
    expected_clean = int(num_patches * 0.25)
    print(f"  Expected clean patches: {expected_clean}, actual clean: {(~mask).sum(dim=1).tolist()}")

    # Test apply_patch_noise
    mixed, noise_out, mask_out = diffusion.apply_patch_noise(patches, mask, t)
    print(f"  Mixed patches shape: {mixed.shape}")
    # Verify clean patches are unchanged
    clean_diff = (mixed[~mask] - patches[~mask]).abs().max().item()
    print(f"  Max diff on clean patches (should be 0): {clean_diff:.6f}")
    print("  [PASS] PatchDiffusion OK\n")


def test_curriculum():
    print("=" * 60)
    print("[3/5] Testing CurriculumScheduler")
    print("=" * 60)
    scheduler = CurriculumScheduler(
        total_epochs=300,
        t_min_start=800, t_min_end=100,
        t_max_start=1000, t_max_end=600,
        clean_ratio_start=0.25, clean_ratio_end=0.05,
        warmup_epochs=10, schedule='cosine',
    )
    print(f"  Scheduler: {scheduler}")

    for epoch in [0, 5, 10, 50, 150, 299]:
        state = scheduler.get_state(epoch)
        print(f"  Epoch {epoch:>3d}: t=[{state['t_min']}, {state['t_max']}], "
              f"clean_ratio={state['clean_ratio']:.3f}, decay={state['decay_factor']:.3f}")

    # Verify monotonic decay
    states = [scheduler.get_state(e) for e in range(300)]
    t_mins = [s['t_min'] for s in states]
    clean_ratios = [s['clean_ratio'] for s in states]
    assert t_mins[0] >= t_mins[-1], "t_min should decay"
    assert clean_ratios[0] >= clean_ratios[-1], "clean_ratio should decay"
    print("  [PASS] CurriculumScheduler OK\n")


def test_model_forward(cfg, device):
    print("=" * 60)
    print("[4/5] Testing SubDiff model forward + backward")
    print("=" * 60)
    curriculum_cfg = {
        't_min_start': cfg['curriculum']['t_min_start'],
        't_min_end': cfg['curriculum']['t_min_end'],
        't_max_start': cfg['curriculum']['t_max_start'],
        't_max_end': cfg['curriculum']['t_max_end'],
        'clean_ratio_start': cfg['curriculum']['clean_ratio_start'],
        'clean_ratio_end': cfg['curriculum']['clean_ratio_end'],
        'warmup_epochs': cfg['curriculum']['warmup_epochs'],
        'schedule': cfg['curriculum']['schedule'],
    }

    model = SubDiff(
        img_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        embed_dim=cfg['model']['embed_dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        decoder_dim=cfg['model']['decoder_embed_dim'],
        decoder_depth=cfg['model']['decoder_depth'],
        decoder_num_heads=cfg['model']['decoder_num_heads'],
        num_timesteps=cfg['diffusion']['num_timesteps'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule_type=cfg['diffusion']['schedule_type'],
        total_epochs=cfg['training']['epochs'],
        curriculum_cfg=curriculum_cfg,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {param_count:.1f}M")

    # Forward pass with a small batch
    dummy_imgs = torch.randn(2, 3, 224, 224, device=device)

    t0 = time.time()
    loss, log_dict = model(dummy_imgs, epoch=0)
    fwd_time = time.time() - t0
    print(f"  Forward pass (epoch=0): loss={loss.item():.4f}, time={fwd_time:.3f}s")
    print(f"  Log: {log_dict}")

    # Backward pass
    t0 = time.time()
    loss.backward()
    bwd_time = time.time() - t0
    print(f"  Backward pass: time={bwd_time:.3f}s")

    # Check gradients exist
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params}")

    # Test at different epochs
    for epoch in [0, 50, 150, 299]:
        model.zero_grad()
        loss, log_dict = model(dummy_imgs, epoch=epoch)
        print(f"  Epoch {epoch:>3d}: loss={loss.item():.4f}, "
              f"t=[{log_dict['t_min']},{log_dict['t_max']}], "
              f"clean_ratio={log_dict['clean_ratio']:.3f}")

    # Test encoder extraction
    encoder = model.get_encoder()
    cls_token, patch_tokens = encoder(dummy_imgs)
    print(f"  Encoder output: cls={cls_token.shape}, patches={patch_tokens.shape}")
    print("  [PASS] SubDiff model OK\n")
    return model


def test_model_with_real_data(cfg, device):
    print("=" * 60)
    print("[5/5] Testing model with real ImageNet data")
    print("=" * 60)
    curriculum_cfg = {
        't_min_start': cfg['curriculum']['t_min_start'],
        't_min_end': cfg['curriculum']['t_min_end'],
        't_max_start': cfg['curriculum']['t_max_start'],
        't_max_end': cfg['curriculum']['t_max_end'],
        'clean_ratio_start': cfg['curriculum']['clean_ratio_start'],
        'clean_ratio_end': cfg['curriculum']['clean_ratio_end'],
        'warmup_epochs': cfg['curriculum']['warmup_epochs'],
        'schedule': cfg['curriculum']['schedule'],
    }

    model = SubDiff(
        img_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        embed_dim=cfg['model']['embed_dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        decoder_dim=cfg['model']['decoder_embed_dim'],
        decoder_depth=cfg['model']['decoder_depth'],
        decoder_num_heads=cfg['model']['decoder_num_heads'],
        num_timesteps=cfg['diffusion']['num_timesteps'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule_type=cfg['diffusion']['schedule_type'],
        total_epochs=cfg['training']['epochs'],
        curriculum_cfg=curriculum_cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

    loader, _ = build_pretrain_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=8,
        num_workers=2,
        distributed=False,
    )

    model.train()
    print("  Running 3 training steps on real data...")
    for step, (imgs, _) in enumerate(loader):
        if step >= 3:
            break
        imgs = imgs.to(device, non_blocking=True)

        loss, log_dict = model(imgs, epoch=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step}: loss={loss.item():.4f}, "
              f"noisy_loss={log_dict['noisy_loss']:.4f}, "
              f"clean_loss={log_dict['clean_loss']:.4f}, "
              f"t_mean={log_dict['t_mean']:.1f}")

    print("  [PASS] Real data training OK\n")


def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'configs', 'pretrain_vit_b16.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Run tests
    imgs = test_data_loading(cfg)
    test_diffusion(imgs)
    test_curriculum()
    test_model_forward(cfg, device)
    test_model_with_real_data(cfg, device)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
