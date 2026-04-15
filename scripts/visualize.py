"""
Visualization utilities for SubDiff pretraining.

Generates side-by-side comparisons at different curriculum stages:
  - Original clean image
  - Noisy input (what the model sees)
  - Model reconstruction
"""

import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff
from subdiff.data import build_eval_dataloader
from einops import rearrange

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(imgs):
    """Undo ImageNet normalization and clamp to [0, 1]."""
    return (imgs * STD.to(imgs.device) + MEAN.to(imgs.device)).clamp(0, 1)


def save_grid(images_list, titles, save_path, nrow=4):
    """Save a grid of image sets using matplotlib.

    Args:
        images_list: list of (B, C, H, W) tensors
        titles: list of column titles
        save_path: output path
        nrow: number of rows (samples)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ncol = len(images_list)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    if nrow == 1:
        axes = axes[None, :]

    for row in range(nrow):
        for col in range(ncol):
            img = images_list[col][row].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def _visualize_dual(model, imgs, epoch, save_dir, device, n_samples=4):
    """Dual-decoder viz: Original | Noisy Input | Eps-recon | Pix-recon."""
    B = imgs.shape[0]
    img_size = int(model.patch_size * (model.num_patches ** 0.5))

    target_patches = model.patchify(imgs)
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    noisy_mask = model.diffusion.generate_noisy_mask(
        B, model.num_patches, model.clean_ratio, device
    )
    mixed_patches, noise, noisy_mask = model.diffusion.apply_patch_noise(
        target_patches, noisy_mask, t
    )
    mixed_imgs = model.unpatchify(mixed_patches, img_size=img_size)

    cls_token, patch_tokens = model.encoder(mixed_imgs)
    pred_eps = model.decoder(patch_tokens)
    pred_pix = model.decoder_pix(patch_tokens)

    # Reconstruct from predicted noise
    sqrt_alpha = model.diffusion.sqrt_alphas_cumprod[t][:, None, None]
    sqrt_one_minus = model.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None]
    eps_recon_patches = (mixed_patches - sqrt_one_minus * pred_eps) / sqrt_alpha
    eps_recon_imgs = model.unpatchify(eps_recon_patches, img_size=img_size)
    pix_recon_imgs = model.unpatchify(pred_pix, img_size=img_size)

    clean_vis = denormalize(imgs)
    noisy_vis = denormalize(mixed_imgs)
    eps_vis = denormalize(eps_recon_imgs)
    pix_vis = denormalize(pix_recon_imgs)

    titles = [
        "Original",
        f"Noisy Input\n(clean={model.clean_ratio}, t=500)",
        "Eps-decoder Recon",
        "Pix-decoder Recon",
    ]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png")
    save_grid([clean_vis, noisy_vis, eps_vis, pix_vis], titles, save_path, nrow=n_samples)


@torch.no_grad()
def _visualize_mae(model, imgs, epoch, save_dir, device, n_samples=4):
    """MAE-style visualization: original | masked+noisy | reconstruction."""
    B = imgs.shape[0]
    img_size = int(model.patch_size * (model.num_patches ** 0.5))

    # Patchify + add noise at a fixed mid-range t for visualization
    target_patches = model.patchify(imgs)
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    noisy_patches, noise = model.diffusion.add_noise(target_patches, t)
    noisy_imgs = model.unpatchify(noisy_patches, img_size=img_size)

    # Masked encoder + decoder
    cls_token, visible_tokens, ids_restore, mask = model.encoder.forward_masked(
        noisy_imgs, mask_ratio=model.mask_ratio
    )
    pred = model.decoder.forward_masked(visible_tokens, ids_restore)

    # Recover clean image
    if model.predict_noise:
        sqrt_alpha = model.diffusion.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = model.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        pred_patches = (noisy_patches - sqrt_one_minus_alpha * pred) / sqrt_alpha
    else:
        pred_patches = pred
    pred_imgs = model.unpatchify(pred_patches, img_size=img_size)

    # Build masked visualization: gray out masked patches on noisy input
    masked_input_patches = noisy_patches.clone()
    mask_expanded = mask.unsqueeze(-1).bool()  # (B, N, 1)
    masked_input_patches = torch.where(mask_expanded, torch.zeros_like(masked_input_patches),
                                        masked_input_patches)
    masked_input_imgs = model.unpatchify(masked_input_patches, img_size=img_size)

    clean_vis = denormalize(imgs)
    masked_vis = denormalize(masked_input_imgs)
    recon_vis = denormalize(pred_imgs)

    titles = [
        "Original",
        f"Masked+Noisy Input\n(mask_ratio={model.mask_ratio}, t=500)",
        "Reconstruction",
    ]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png")
    save_grid([clean_vis, masked_vis, recon_vis], titles, save_path, nrow=n_samples)


@torch.no_grad()
def visualize_epoch(model, imgs, epoch, save_dir, device, n_samples=4):
    """Visualize model behavior at a given epoch stage.

    Generates: original | noisy input | reconstruction
    """
    model.eval()
    imgs = imgs[:n_samples].to(device)
    B = imgs.shape[0]

    # Dual-decoder pretraining: visualize both noise-derived and pixel recons
    if getattr(model, 'dual_decoder', False):
        _visualize_dual(model, imgs, epoch, save_dir, device, n_samples)
        model.train()
        return

    # MAE-style masked pretraining: use MAE visualization path
    if getattr(model, 'mae_masking', False):
        _visualize_mae(model, imgs, epoch, save_dir, device, n_samples)
        model.train()
        return

    # Get curriculum state
    state = model.curriculum.get_state(epoch)
    t_min, t_max = state['t_min'], state['t_max']
    clean_ratio = state['clean_ratio']

    # Patchify
    target_patches = model.patchify(imgs)

    # Sample timesteps and mask
    t = model.diffusion.sample_timesteps(B, t_min, t_max, device)
    noisy_mask = model.diffusion.generate_noisy_mask(B, model.num_patches, clean_ratio, device)

    # Apply noise
    mixed_patches, noise, noisy_mask = model.diffusion.apply_patch_noise(
        target_patches, noisy_mask, t
    )

    # Build noisy image for encoder
    img_size = int(model.patch_size * (model.num_patches ** 0.5))
    mixed_imgs = model.unpatchify(mixed_patches, img_size=img_size)

    # Encode + Decode
    cls_token, patch_tokens = model.encoder(mixed_imgs)
    pred = model.decoder(patch_tokens)

    if getattr(model, 'predict_noise', False):
        # Noise prediction mode: reconstruct by removing predicted noise
        # x_0 = (x_t - sqrt(1-alpha_cumprod) * pred_noise) / sqrt(alpha_cumprod)
        sqrt_alpha = model.diffusion.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = model.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        pred_patches = (mixed_patches - sqrt_one_minus_alpha * pred) / sqrt_alpha
    else:
        pred_patches = pred

    pred_imgs = model.unpatchify(pred_patches, img_size=img_size)

    # Build mask visualization (highlight noisy patches)
    mask_vis = torch.ones_like(imgs) * 0.3  # dim background
    p = model.patch_size
    h = w = img_size // p
    for b in range(B):
        for idx in range(model.num_patches):
            row, col = idx // w, idx % w
            if noisy_mask[b, idx]:
                # Noisy patch: show in red tint
                mask_vis[b, 0, row*p:(row+1)*p, col*p:(col+1)*p] = 0.8
                mask_vis[b, 1, row*p:(row+1)*p, col*p:(col+1)*p] = 0.2
                mask_vis[b, 2, row*p:(row+1)*p, col*p:(col+1)*p] = 0.2
            else:
                # Clean patch: show in green tint
                mask_vis[b, 0, row*p:(row+1)*p, col*p:(col+1)*p] = 0.2
                mask_vis[b, 1, row*p:(row+1)*p, col*p:(col+1)*p] = 0.8
                mask_vis[b, 2, row*p:(row+1)*p, col*p:(col+1)*p] = 0.2

    # Denormalize for visualization
    clean_vis = denormalize(imgs)
    noisy_vis = denormalize(mixed_imgs)
    recon_vis = denormalize(pred_imgs)

    t_str = f"t=[{t_min},{t_max}]"
    titles = [
        "Original",
        f"Mask (red=noisy, green=clean)\nclean_ratio={clean_ratio:.2f}",
        f"Noisy Input\n{t_str}",
        "Reconstruction",
    ]

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png")
    save_grid([clean_vis, mask_vis, noisy_vis, recon_vis], titles, save_path, nrow=n_samples)

    model.train()


def visualize_from_checkpoint(config_path, checkpoint_path, save_dir, epoch_override=None):
    """Load a checkpoint and produce visualization."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    epoch = epoch_override if epoch_override is not None else ckpt['epoch']
    print(f"Loaded checkpoint epoch {ckpt['epoch']}, visualizing as epoch {epoch}")

    # Get a batch of val images
    val_loader, _ = build_eval_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=8,
        num_workers=2,
    )
    imgs, _ = next(iter(val_loader))

    visualize_epoch(model, imgs, epoch, save_dir, device, n_samples=4)


def visualize_curriculum_stages(config_path, save_dir):
    """Visualize what different curriculum stages look like with a random model.
    No checkpoint needed — shows the noise/mask patterns at different epochs.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Get real images
    val_loader, _ = build_eval_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=8,
        num_workers=2,
    )
    imgs, _ = next(iter(val_loader))

    # Visualize at key curriculum stages
    stages = [0, 10, 50, 100, 150, 200, 250, 299]
    print(f"Visualizing curriculum stages: {stages}")
    for epoch in stages:
        visualize_epoch(model, imgs, epoch, save_dir, device, n_samples=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pretrain_vit_b16.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='If provided, visualize from this checkpoint')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Override epoch for curriculum state')
    parser.add_argument('--save_dir', type=str, default='logs/visualizations')
    parser.add_argument('--stages', action='store_true',
                        help='Visualize all curriculum stages (no checkpoint needed)')
    args = parser.parse_args()

    if args.stages:
        visualize_curriculum_stages(args.config, args.save_dir)
    elif args.checkpoint:
        visualize_from_checkpoint(args.config, args.checkpoint, args.save_dir, args.epoch)
    else:
        print("Provide --checkpoint or --stages")
