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


def _noise_to_vis(noise_img):
    """Visualize a noise-map tensor (any range) as a [0, 1] image.
    ε̂ is ~N(0, 1), so scale by 0.3 + 0.5 maps ±2σ region into the viewable range.
    """
    return (noise_img * 0.3 + 0.5).clamp(0, 1)


def _ddpm_one_step_reverse(model, noisy_patches, pred_noise_patches, t):
    """Compute x_{t-1} via the standard DDPM posterior mean, deterministically
    (no stochastic noise injection). This is what the model actually produces
    in one DDPM reverse step, given its ε̂ prediction.

    x_{t-1} should be only slightly less noisy than x_t, NOT fully clean.
    """
    diff = model.diffusion
    alpha_bar_t = diff.alphas_cumprod[t][:, None, None]
    sqrt_alpha_bar_t = alpha_bar_t.sqrt()
    sqrt_one_minus_t = (1 - alpha_bar_t).sqrt()

    # Predicted clean x_0 from ε̂ (intermediate, used in posterior formula)
    pred_x0 = (noisy_patches - sqrt_one_minus_t * pred_noise_patches) / sqrt_alpha_bar_t
    pred_x0 = pred_x0.clamp(-3, 3)

    t_prev = (t - 1).clamp(min=0)
    alpha_bar_prev = diff.alphas_cumprod[t_prev][:, None, None]
    beta_t = diff.betas[t][:, None, None]
    alpha_t = 1.0 - beta_t

    posterior_mean = (
        alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t) * pred_x0
        + alpha_t.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * noisy_patches
    )
    return posterior_mean, pred_x0


@torch.no_grad()
def _visualize_naive_ddpm(model, imgs, epoch, save_dir, device, n_samples=4):
    """Naive DDPM-ViT viz — shows the model's actual single-step behavior:
      col 1: Original x_0 (ground truth)
      col 2: Noisy input x_t (at t=500)
      col 3: ε̂ (model's raw predicted noise, visualized as a map)
      col 4: x_{t-1} — one DDPM reverse step (slightly denoised, NOT fully clean)
      col 5: pred_x̂_0 — model's "optimistic" estimate if ε̂ were perfect
             (what the OLD visualization incorrectly showed as main output)
    """
    B = imgs.shape[0]
    img_size = int(model.patch_size * (model.num_patches ** 0.5))

    target_patches = model.patchify(imgs)
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    noisy_patches, _ = model.diffusion.add_noise(target_patches, t)
    noisy_imgs = model.unpatchify(noisy_patches, img_size=img_size)

    cls_token, patch_tokens = model._encode_with_time(noisy_imgs, t)
    pred_noise_patches = model.decoder(patch_tokens)
    pred_noise_patches = model._apply_conv_refine(pred_noise_patches)

    x_prev_patches, pred_x0_patches = _ddpm_one_step_reverse(
        model, noisy_patches, pred_noise_patches, t)

    pred_noise_img = model.unpatchify(pred_noise_patches, img_size=img_size)
    x_prev_img = model.unpatchify(x_prev_patches, img_size=img_size)
    pred_x0_img = model.unpatchify(pred_x0_patches, img_size=img_size)

    clean_vis = denormalize(imgs)
    noisy_vis = denormalize(noisy_imgs)
    eps_vis = _noise_to_vis(pred_noise_img)
    xprev_vis = denormalize(x_prev_img)
    x0hat_vis = denormalize(pred_x0_img)

    titles = [
        "Original x_0",
        "Noisy x_t (t=500)",
        "Predicted ε̂",
        "x_{t-1} (one DDPM step)",
        "x̂_0 (lookahead)",
    ]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png")
    save_grid([clean_vis, noisy_vis, eps_vis, xprev_vis, x0hat_vis],
              titles, save_path, nrow=n_samples)


@torch.no_grad()
def _visualize_naive_rf(model, imgs, epoch, save_dir, device, n_samples=4):
    """Rectified Flow (SD3/FLUX) viz — model predicts v = ε − x_0.

    RF path: x_t = (1−t)·x_0 + t·ε with t ∈ (0, 1).
    Algebraic decomposition of v̂ (no sampler iterations needed):
      x̂_0 = x_t − t·v̂        (clean lookahead, also the single Euler step to t=0)
      ε̂   = x_t + (1−t)·v̂    (noise recovery)

    Columns:
      1. Original x_0
      2. Noisy x_t (t=0.5 — half signal, half noise)
      3. Predicted v̂ (as mean-centered map; v ~ N(0, 2) per pixel a priori)
      4. Recovered ε̂
      5. x̂_0 (lookahead; equivalently one Euler step all the way to t=0)

    Note: for RF+MAE (Run 10), this runs at r=0 (no mask) — the inference
    distribution, matching the r≈0 tail of training.
    """
    B = imgs.shape[0]
    img_size = int(model.patch_size * (model.num_patches ** 0.5))

    # Fixed vis-time t=0.5: middle of the RF path (signal and noise ~equal weight).
    t_cont = torch.full((B,), 0.5, device=device)
    t_b = t_cont.view(B, 1, 1)

    target_patches = model.patchify(imgs)
    eps = torch.randn_like(target_patches)
    x_t_patches = (1 - t_b) * target_patches + t_b * eps
    x_t_imgs = model.unpatchify(x_t_patches, img_size=img_size)

    # Model forward (integer t for sinusoidal embed, consistent with training path)
    t_int = (t_cont * (model.diffusion.num_timesteps - 1)).long()
    cls_token, patch_tokens = model._encode_with_time(x_t_imgs, t_int)
    pred_v_patches = model.decoder(patch_tokens)
    pred_v_patches = model._apply_conv_refine(pred_v_patches)

    # Algebraic decomposition (no sampler)
    pred_x0_patches = x_t_patches - t_b * pred_v_patches
    pred_eps_patches = x_t_patches + (1 - t_b) * pred_v_patches

    v_img = model.unpatchify(pred_v_patches, img_size=img_size)
    eps_img = model.unpatchify(pred_eps_patches, img_size=img_size)
    x0_img = model.unpatchify(pred_x0_patches, img_size=img_size)

    clean_vis = denormalize(imgs)
    noisy_vis = denormalize(x_t_imgs)
    # v has ~√2 std, scale slightly tighter than ε̂
    v_vis = (v_img * 0.2 + 0.5).clamp(0, 1)
    eps_vis = _noise_to_vis(eps_img)
    x0_vis = denormalize(x0_img)

    titles = [
        "Original x_0",
        "Noisy x_t (t=0.5)",
        "Predicted v̂",
        "Recovered ε̂",
        "x̂_0 (Euler → t=0)",
    ]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png")
    save_grid([clean_vis, noisy_vis, v_vis, eps_vis, x0_vis],
              titles, save_path, nrow=n_samples)


@torch.no_grad()
def _visualize_naive_mae(model, imgs, epoch, save_dir, device, n_samples=4):
    """Naive MAE viz: Original | Masked Input | Reconstruction."""
    B = imgs.shape[0]
    img_size = int(model.patch_size * (model.num_patches ** 0.5))

    cls_token, visible_tokens, ids_restore, mask = model.encoder.forward_masked(
        imgs, mask_ratio=model.mask_ratio
    )
    pred_patches = model.decoder.forward_masked(visible_tokens, ids_restore)

    # Build "masked input" = clean visible, gray for masked
    target_patches = model.patchify(imgs)
    mask_expanded = mask.unsqueeze(-1).bool()
    masked_input_patches = torch.where(mask_expanded, torch.zeros_like(target_patches),
                                        target_patches)
    masked_input_imgs = model.unpatchify(masked_input_patches, img_size=img_size)
    recon_imgs = model.unpatchify(pred_patches, img_size=img_size)

    clean_vis = denormalize(imgs)
    masked_vis = denormalize(masked_input_imgs)
    recon_vis = denormalize(recon_imgs)

    titles = [
        "Original",
        f"Masked Input (mask_ratio={model.mask_ratio})",
        "Reconstruction",
    ]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png")
    save_grid([clean_vis, masked_vis, recon_vis], titles, save_path, nrow=n_samples)


@torch.no_grad()
def _visualize_dual(model, imgs, epoch, save_dir, device, n_samples=4):
    """Dual-decoder viz — shows one single-step reverse (not lookahead):
      col 1: Original x_0
      col 2: Noisy Input x_t (25% clean + 75% noisy, t=500)
      col 3: ε̂ (eps-decoder raw output)
      col 4: x_{t-1} (one DDPM reverse step from ε head)
      col 5: Pix-decoder direct output (x_0 prediction from pix head)
    """
    B = imgs.shape[0]
    img_size = int(model.patch_size * (model.num_patches ** 0.5))

    target_patches = model.patchify(imgs)
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    noisy_mask = model.diffusion.generate_noisy_mask(
        B, model.num_patches, model.clean_ratio, device
    )
    mixed_patches, _, noisy_mask = model.diffusion.apply_patch_noise(
        target_patches, noisy_mask, t
    )
    mixed_imgs = model.unpatchify(mixed_patches, img_size=img_size)

    if getattr(model, '_use_dit_encoder', False):
        cls_token, patch_tokens = model._encode_with_indicators(mixed_imgs, noisy_mask, t)
    else:
        cls_token, patch_tokens = model.encoder(mixed_imgs)
    pred_eps = model.decoder(patch_tokens)
    pred_pix = model.decoder_pix(patch_tokens)
    pred_eps = model._apply_conv_refine(pred_eps)
    pred_pix = model._apply_conv_refine(pred_pix)

    # One-step DDPM reverse using ε head
    x_prev_patches, _ = _ddpm_one_step_reverse(model, mixed_patches, pred_eps, t)
    x_prev_img = model.unpatchify(x_prev_patches, img_size=img_size)
    pred_eps_img = model.unpatchify(pred_eps, img_size=img_size)
    pred_pix_img = model.unpatchify(pred_pix, img_size=img_size)

    clean_vis = denormalize(imgs)
    noisy_vis = denormalize(mixed_imgs)
    eps_map_vis = _noise_to_vis(pred_eps_img)
    xprev_vis = denormalize(x_prev_img)
    pix_vis = denormalize(pred_pix_img)

    titles = [
        "Original x_0",
        f"Noisy x_t\n(clean={model.clean_ratio}, t=500)",
        "Predicted ε̂",
        "x_{t-1} (one DDPM step)",
        "Pix-decoder output (x̂_0)",
    ]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png")
    save_grid([clean_vis, noisy_vis, eps_map_vis, xprev_vis, pix_vis],
              titles, save_path, nrow=n_samples)


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

    # Rectified Flow (v-pred) — check BEFORE naive_ddpm because RF+MAE has
    # both flags set (naive_ddpm=True, flow_matching=True)
    if getattr(model, 'flow_matching', False):
        _visualize_naive_rf(model, imgs, epoch, save_dir, device, n_samples)
        model.train()
        return

    # Naive DDPM-ViT (ε-pred)
    if getattr(model, 'naive_ddpm', False):
        _visualize_naive_ddpm(model, imgs, epoch, save_dir, device, n_samples)
        model.train()
        return

    # Naive MAE
    if getattr(model, 'naive_mae', False):
        _visualize_naive_mae(model, imgs, epoch, save_dir, device, n_samples)
        model.train()
        return

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

    # Encode + Decode (route on encoder type: DiT needs time conditioning)
    if getattr(model, '_use_dit_encoder', False):
        cls_token, patch_tokens = model._encode_with_indicators(mixed_imgs, noisy_mask, t)
    else:
        cls_token, patch_tokens = model.encoder(mixed_imgs)
    pred = model.decoder(patch_tokens)
    pred = model._apply_conv_refine(pred)

    predict_noise = getattr(model, 'predict_noise', False)

    # Build mask visualization (highlight noisy patches)
    mask_vis = torch.ones_like(imgs) * 0.3
    p = model.patch_size
    h = w = img_size // p
    for b in range(B):
        for idx in range(model.num_patches):
            row, col = idx // w, idx % w
            if noisy_mask[b, idx]:
                mask_vis[b, 0, row*p:(row+1)*p, col*p:(col+1)*p] = 0.8
                mask_vis[b, 1, row*p:(row+1)*p, col*p:(col+1)*p] = 0.2
                mask_vis[b, 2, row*p:(row+1)*p, col*p:(col+1)*p] = 0.2
            else:
                mask_vis[b, 0, row*p:(row+1)*p, col*p:(col+1)*p] = 0.2
                mask_vis[b, 1, row*p:(row+1)*p, col*p:(col+1)*p] = 0.8
                mask_vis[b, 2, row*p:(row+1)*p, col*p:(col+1)*p] = 0.2

    clean_vis = denormalize(imgs)
    noisy_vis = denormalize(mixed_imgs)
    t_str = f"t=[{t_min},{t_max}]"

    if predict_noise:
        # ε-prediction path: show ε̂ and one-step DDPM reverse (not lookahead).
        x_prev_patches, _ = _ddpm_one_step_reverse(model, mixed_patches, pred, t)
        x_prev_img = model.unpatchify(x_prev_patches, img_size=img_size)
        pred_eps_img = model.unpatchify(pred, img_size=img_size)

        eps_vis = _noise_to_vis(pred_eps_img)
        xprev_vis = denormalize(x_prev_img)
        titles = [
            "Original",
            f"Mask (red=noisy, green=clean)\nclean_ratio={clean_ratio:.2f}",
            f"Noisy Input\n{t_str}",
            "Predicted ε̂",
            "x_{t-1} (one DDPM step)",
        ]
        save_grid([clean_vis, mask_vis, noisy_vis, eps_vis, xprev_vis],
                  titles, os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png"),
                  nrow=n_samples)
    else:
        # pixel-reconstruction path: pred IS x_0 directly
        pred_imgs = model.unpatchify(pred, img_size=img_size)
        recon_vis = denormalize(pred_imgs)
        titles = [
            "Original",
            f"Mask (red=noisy, green=clean)\nclean_ratio={clean_ratio:.2f}",
            f"Noisy Input\n{t_str}",
            "Pixel Reconstruction",
        ]
        save_grid([clean_vis, mask_vis, noisy_vis, recon_vis],
                  titles, os.path.join(save_dir, f"vis_epoch_{epoch:04d}.png"),
                  nrow=n_samples)

    os.makedirs(save_dir, exist_ok=True)
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
