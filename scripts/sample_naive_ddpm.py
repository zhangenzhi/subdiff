"""
Sample images from a naive DDPM-ViT checkpoint.

Standard DDPM ancestral sampling (DDPM, Ho et al. 2020):
  x_T ~ N(0, I)
  for t = T-1, ..., 0:
    eps_pred = model(x_t, t)
    mean = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_t)
    if t > 0:
      x_{t-1} = mean + sqrt(beta_t) * z, z ~ N(0, I)
    else:
      x_0 = mean

Usage:
  python scripts/sample_naive_ddpm.py \
      --config configs/pretrain_vit_b16_naive_ddpm.yaml \
      --checkpoint logs_naive_ddpm/checkpoints/checkpoint_latest.pth \
      --num_samples 16 \
      --output_dir samples_naive_ddpm/
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff


# ImageNet normalization (training-time)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--num_samples', type=int, default=16)
    p.add_argument('--num_steps', type=int, default=None,
                   help='DDPM sampling steps (default: num_timesteps from config)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--output_dir', type=str, default='samples/')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eta', type=float, default=0.0,
                   help='DDIM eta: 0=deterministic, 1=DDPM stochastic (default 0)')
    return p.parse_args()


def load_model(cfg, ckpt_path, device):
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
        total_epochs=1,
        predict_noise=cfg.get('diffusion', {}).get('predict_noise', False),
        dual_decoder=cfg['model'].get('dual_decoder', False),
        clean_ratio=cfg['model'].get('clean_ratio', 0.25),
        naive_ddpm=cfg['model'].get('naive_ddpm', False),
        qk_norm=cfg['model'].get('qk_norm', False),
        dit_minimal_head=cfg['model'].get('dit_minimal_head', False),
        use_indicators=cfg['model'].get('use_indicators', False),
        use_conv_refine=cfg['model'].get('use_conv_refine', False),
        loss_weighting=cfg.get('diffusion', {}).get('loss_weighting', 'simple'),
        snr_gamma=cfg.get('diffusion', {}).get('snr_gamma', 5.0),
        pos_embed_type=cfg['model'].get('pos_embed_type', 'sincos'),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # Skip loading pos_embed when using fixed sin-cos (requires_grad=False):
    # otherwise the checkpoint's (learned) pos_embed would overwrite our
    # fresh sin-cos values.
    state_to_load = ckpt['model']
    if getattr(model.encoder, 'pos_embed_type', 'learnable') == 'sincos':
        state_to_load = {k: v for k, v in state_to_load.items()
                         if not k.endswith('encoder.pos_embed')}
    model.load_state_dict(state_to_load, strict=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"avg_loss={ckpt.get('avg_loss', 'n/a')}")
    model.eval()
    return model


def _predict_noise(model, x, t):
    """Dispatch noise prediction based on model mode.

    - naive_ddpm: encoder is DiTEncoder with adaLN; uses _encode_with_time
    - SubDiff (dual / eps_qknorm / orig with predict_noise):
      uses _encode_with_indicators with all-noisy mask + time embedding
    """
    img_size = x.shape[-1]
    if model.naive_ddpm:
        cls_token, patch_tokens = model._encode_with_time(x, t)
    else:
        # SubDiff path: at sampling time, no clean anchors are available
        # (we're generating from pure noise). Mark all patches as noisy.
        B = x.shape[0]
        noisy_mask = torch.ones(B, model.num_patches, dtype=torch.bool, device=x.device)
        cls_token, patch_tokens = model._encode_with_indicators(x, noisy_mask, t)
    pred_noise_patches = model.decoder(patch_tokens)
    pred_noise_patches = model._apply_conv_refine(pred_noise_patches)
    return model.unpatchify(pred_noise_patches, img_size=img_size)


@torch.no_grad()
def ddpm_sample(model, batch_size, img_size, num_steps, device, eta=0.0):
    """DDIM-style sampling (correct for any stride).

    DDIM update rule (eta=0 is deterministic, eta=1 recovers DDPM):
        x_{t_prev} = sqrt(alpha_bar_prev) * x0_hat
                     + sqrt(1 - alpha_bar_prev - sigma^2) * eps_hat
                     + sigma * eps
    where sigma = eta * sqrt((1-alpha_bar_prev)/(1-alpha_bar_t) * (1-alpha_bar_t/alpha_bar_prev))

    Args:
        model: SubDiff model (predicts noise)
        batch_size, img_size: output shape
        num_steps: number of denoising steps (can be << T, strided sampling)
        eta: stochasticity (0 = DDIM deterministic, 1 = DDPM full noise)

    Returns:
        x_0: (B, 3, H, W) generated images in normalized space
    """
    diff = model.diffusion
    T = diff.num_timesteps
    if num_steps is None:
        num_steps = T

    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # Evenly-spaced integer timesteps from T-1 down to 0
    ts = torch.linspace(T - 1, 0, num_steps, dtype=torch.long, device=device).tolist()

    for i, t_cur in enumerate(ts):
        t = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)
        pred_noise = _predict_noise(model, x, t)

        alpha_bar_t = diff.alphas_cumprod[t_cur]
        sqrt_alpha_bar = alpha_bar_t.sqrt()
        sqrt_one_minus = (1 - alpha_bar_t).sqrt()

        # Estimate x_0 from current x_t and predicted noise
        pred_x0 = (x - sqrt_one_minus * pred_noise) / sqrt_alpha_bar
        pred_x0 = pred_x0.clamp(-3, 3)  # ImageNet-normalized range

        if i < len(ts) - 1:
            t_prev = ts[i + 1]
            alpha_bar_prev = diff.alphas_cumprod[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        # DDIM stochasticity term (eta=0 → deterministic)
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t)
                       * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
        # Direction pointing to x_t (using predicted noise)
        dir_xt = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise

        x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt
        if eta > 0 and i < len(ts) - 1:
            x = x + sigma * torch.randn_like(x)

        if (i + 1) % max(1, num_steps // 10) == 0 or i == num_steps - 1:
            print(f"  step {i+1}/{num_steps}  t={t_cur}  "
                  f"x range=[{x.min().item():.2f}, {x.max().item():.2f}]")

    return x


def denormalize(imgs):
    """Undo ImageNet normalization, clamp to [0, 1]."""
    return (imgs * STD.to(imgs.device) + MEAN.to(imgs.device)).clamp(0, 1)


def save_grid(imgs, path, nrow=4):
    """Save a grid of images using matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = imgs.shape[0]
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(3 * ncol, 3 * nrow_actual))
    if nrow_actual == 1:
        axes = axes[None, :]

    for i in range(nrow_actual * ncol):
        ax = axes[i // ncol, i % ncol]
        if i < n:
            img = imgs[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = get_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(cfg, args.checkpoint, device)
    img_size = cfg['data']['image_size']

    os.makedirs(args.output_dir, exist_ok=True)

    all_samples = []
    n_done = 0
    while n_done < args.num_samples:
        bs = min(args.batch_size, args.num_samples - n_done)
        print(f"\nGenerating batch of {bs} samples ({n_done}/{args.num_samples})...")
        x_norm = ddpm_sample(model, bs, img_size, args.num_steps, device, eta=args.eta)
        x_vis = denormalize(x_norm)
        all_samples.append(x_vis)
        n_done += bs

    samples = torch.cat(all_samples, dim=0)

    grid_path = os.path.join(args.output_dir, 'grid.png')
    save_grid(samples, grid_path, nrow=4)
    print(f"\nSaved grid: {grid_path}")

    # Also save individual images
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i, img in enumerate(samples):
        path = os.path.join(args.output_dir, f'sample_{i:03d}.png')
        plt.imsave(path, img.cpu().permute(1, 2, 0).numpy())
    print(f"Saved {len(samples)} individual images to {args.output_dir}")


if __name__ == '__main__':
    main()
