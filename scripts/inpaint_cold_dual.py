"""
Run 12 cold-RF inpaint: K-step Euler from x_t=μ to x_t=x_0.

Compares Run X single-pass (μ from x_0-head, what we already had) against
Run 12's iterative refinement at K = 1, 4, 8, 16, ..., on the SAME seed,
SAME images, SAME mask as the Run X t=1 inpaint test — so the resulting
grid is directly stackable next to samples_rf_p8_inpaint_t1/t1.0_p0.25_ep299.png.

Usage:
  python scripts/inpaint_cold_dual.py \
    --mu_config configs/pretrain_vit_b8_dual_rf.yaml \
    --mu_ckpt   logs_dual_rf_p8/checkpoints/checkpoint_latest.pth \
    --refiner_config configs/pretrain_vit_b8_cold_rf.yaml \
    --refiner_ckpt   logs_run12_cold_rf/checkpoints/checkpoint_latest.pth \
    --Ks 1,4,8,16 --num_samples 4 \
    --output_dir samples_run12_inpaint/
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.data import build_eval_dataloader
from scripts.pretrain import build_subdiff_from_cfg

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mu_config', required=True)
    p.add_argument('--mu_ckpt', required=True)
    p.add_argument('--refiner_config', required=True)
    p.add_argument('--refiner_ckpt', required=True)
    p.add_argument('--prompt_ratio', type=float, default=0.25)
    p.add_argument('--num_samples', type=int, default=4)
    p.add_argument('--Ks', type=str, default='1,4,8,16')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='samples_run12_inpaint/')
    return p.parse_args()


def _build(cfg_path, ckpt_path, device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    curriculum_cfg = {k: cfg['curriculum'][k] for k in
                      ['t_min_start', 't_min_end', 't_max_start', 't_max_end',
                       'clean_ratio_start', 'clean_ratio_end',
                       'warmup_epochs', 'schedule']}
    model = build_subdiff_from_cfg(cfg, curriculum_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model']
    if getattr(model.encoder, 'pos_embed_type', 'learnable') == 'sincos':
        state = {k: v for k, v in state.items()
                 if not k.endswith('encoder.pos_embed')}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  loaded {cfg_path} @ ep={ckpt.get('epoch')} "
          f"avg_loss={ckpt.get('avg_loss', 'n/a')}")
    return model, cfg


def denormalize(imgs):
    return (imgs * STD.to(imgs.device) + MEAN.to(imgs.device)).clamp(0, 1)


def save_grid(images_list, titles, save_path, nrow):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncol = len(images_list)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow))
    if nrow == 1:
        axes = axes[None, :]
    for r in range(nrow):
        for c in range(ncol):
            img = images_list[c][r].cpu().permute(1, 2, 0).numpy()
            axes[r, c].imshow(np.clip(img, 0, 1))
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


@torch.no_grad()
def refiner_pred_v(refiner, mixed_imgs, t_value, noisy_mask):
    """One forward pass through refiner; returns pred_v on all positions."""
    B = mixed_imgs.shape[0]
    device = mixed_imgs.device
    t_cont = torch.full((B,), t_value, device=device)
    t_int = (t_cont * (refiner.diffusion.num_timesteps - 1)).long()
    cls_token, patch_tokens = refiner._encode_with_indicators(
        mixed_imgs, noisy_mask, t_int)
    pred_v = refiner.decoder(patch_tokens)
    pred_v = refiner._apply_conv_refine(pred_v)
    return pred_v


@torch.no_grad()
def cold_euler_inpaint(refiner, mu, target_patches, noisy_mask, K, img_size):
    """K-step Euler from x_t=μ at t=1 to x_t=x_0 at t=0.
       Clean positions stay at target_patches throughout.
       Returns final composite patches (B, N, D)."""
    m = noisy_mask.unsqueeze(-1).float()
    x_t_noisy = mu.clone()
    for k in range(K):
        t_curr = 1.0 - k / K
        mixed = m * x_t_noisy + (1 - m) * target_patches
        mixed_imgs = refiner.unpatchify(mixed, img_size=img_size)
        pred_v = refiner_pred_v(refiner, mixed_imgs, t_curr, noisy_mask)
        x_t_noisy = x_t_noisy - (1.0 / K) * pred_v
    final = m * x_t_noisy + (1 - m) * target_patches
    return final


@torch.no_grad()
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("Building μ-generator (Run X)...")
    mu_model, mu_cfg = _build(args.mu_config, args.mu_ckpt, device)
    print("Building Refiner (Run 12)...")
    refiner, ref_cfg = _build(args.refiner_config, args.refiner_ckpt, device)

    # Same val images as inpaint_t1_grid.py (seed + first batch)
    val_loader, _ = build_eval_dataloader(
        imagenet_dir=mu_cfg['data']['imagenet_dir'],
        image_size=mu_cfg['data']['image_size'],
        batch_size=args.num_samples, num_workers=2, backend='torch',
    )
    imgs, _ = next(iter(val_loader))
    imgs = imgs.to(device)
    B = imgs.shape[0]
    N = mu_model.num_patches
    img_size = int(mu_model.patch_size * (N ** 0.5))
    target_patches = mu_model.patchify(imgs)

    # 25% clean prompt mask — same RNG path as inpaint_t1_grid.py
    rand = torch.rand(B, N, device=device)
    k = int(round(N * args.prompt_ratio))
    ids_keep = torch.argsort(rand, dim=1)[:, :k]
    prompt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    prompt_mask.scatter_(1, ids_keep, True)
    noisy_mask = ~prompt_mask
    m = noisy_mask.unsqueeze(-1).float()

    # μ from frozen mu_model (Run X x_0-head, t=1, pure noise at noisy)
    mu = mu_model.compute_mu(imgs, noisy_mask)

    # ---- Build viz columns ----
    cols, titles = [], []

    orig = denormalize(imgs)
    cols.append(orig); titles.append('Original x_0')

    gray = torch.zeros_like(target_patches)
    clean_only = (1 - m) * target_patches + m * gray
    cols.append(denormalize(mu_model.unpatchify(clean_only, img_size=img_size)))
    titles.append(f'Clean prompt {int(args.prompt_ratio*100)}%')

    # Run X single pass = μ at noisy + true x_0 at clean (composite)
    runx_composite = (1 - m) * target_patches + m * mu
    cols.append(denormalize(mu_model.unpatchify(runx_composite, img_size=img_size)))
    titles.append('Run X (μ, single)')

    # ----- metrics ----------------------------------------------------------
    # pixel-MSE rewards mean prediction (μ wins because it's already the
    # conditional mean). For inpaint quality we also want high-frequency
    # fidelity — apply a Laplacian high-pass and report its MSE.
    def _laplacian(x):
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                              dtype=x.dtype, device=x.device)
        kernel = kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def _hf_metrics(pred_imgs, true_imgs, mask_img):
        """Returns (hf_mse, hf_energy_pred, hf_energy_target) — all averaged
        over masked pixels and color channels."""
        lap_p = _laplacian(pred_imgs)
        lap_t = _laplacian(true_imgs)
        sq_diff = ((lap_p - lap_t) ** 2).mean(dim=1, keepdim=True) * mask_img
        sq_p = (lap_p ** 2).mean(dim=1, keepdim=True) * mask_img
        sq_t = (lap_t ** 2).mean(dim=1, keepdim=True) * mask_img
        denom = mask_img.sum().clamp(min=1.0)
        return (sq_diff.sum() / denom).item(), \
               (sq_p.sum() / denom).item(), \
               (sq_t.sum() / denom).item()

    # Build pixel-space mask (B,1,H,W) — 1 at noisy pixels.
    mask_img = refiner.unpatchify(m.expand(-1, -1, target_patches.shape[-1]),
                                   img_size=img_size)
    # Take any one channel (they're identical because m repeated)
    mask_img = mask_img[:, :1]

    target_imgs_norm = imgs  # in normalized space (matches mu/x_t outputs)

    # Run 12 K-step Euler for each K
    Ks = [int(x) for x in args.Ks.split(',')]
    print(f"\n=== noisy-position metrics (B={B}) ===")

    mu_mse = ((mu - target_patches) ** 2).mean(dim=-1)[noisy_mask].mean().item()
    mu_imgs = mu_model.unpatchify(
        (1 - m) * target_patches + m * mu, img_size=img_size)
    mu_hf_mse, mu_hf_e, target_hf_e = _hf_metrics(mu_imgs, target_imgs_norm, mask_img)
    print(f"  {'Method':<22}{'pixel_MSE':>11}{'HF_MSE':>11}{'HF_energy':>11}"
          f"{'HF_e/target':>13}")
    print(f"  {'Target (reference)':<22}{0.0:>11.5f}{0.0:>11.5f}"
          f"{target_hf_e:>11.5f}{1.000:>13.3f}")
    print(f"  {'Run X μ (single)':<22}{mu_mse:>11.5f}{mu_hf_mse:>11.5f}"
          f"{mu_hf_e:>11.5f}{mu_hf_e/target_hf_e:>13.3f}")

    for K in Ks:
        comp = cold_euler_inpaint(
            refiner, mu, target_patches, noisy_mask, K, img_size)
        cols.append(denormalize(refiner.unpatchify(comp, img_size=img_size)))
        titles.append(f'Run 12 K={K}')

        k_mse = ((comp - target_patches) ** 2).mean(dim=-1)[noisy_mask].mean().item()
        comp_imgs = refiner.unpatchify(comp, img_size=img_size)
        k_hf_mse, k_hf_e, _ = _hf_metrics(comp_imgs, target_imgs_norm, mask_img)
        print(f"  {'Run 12 K=' + str(K):<22}{k_mse:>11.5f}{k_hf_mse:>11.5f}"
              f"{k_hf_e:>11.5f}{k_hf_e/target_hf_e:>13.3f}")

    save_grid(
        cols, titles,
        os.path.join(args.output_dir, f"cold_t1_p{args.prompt_ratio}.png"),
        nrow=B,
    )


if __name__ == '__main__':
    main()
