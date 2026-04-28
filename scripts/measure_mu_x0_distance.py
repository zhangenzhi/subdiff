"""
Step 1 for Run Y (cold-RF):  measure how far Run X's x_0-head output (μ)
is from true x_0 at noisy positions, when the input is the inpaint setup
(25% clean prompt + 75% pure noise at t=1).

This decides whether the cold-diffusion `mean → x_0` chain has enough
residual for a Refiner v-head to learn anything.

Reports per-noisy-patch MSE statistics, against two baselines:
  - Predict-zero (= predicting per-pixel mean, since data is normalized):
    MSE = E[x_0^2] ≈ 1.0
  - Training x_0-head loss (avg over t ~ logit_normal): looked up from
    training log, ≈ 0.027 at ep 269

If MSE(μ, x_0) at t=1.0 >> 0.01, the chain is non-trivial → Refiner has
work to do. If < 0.01, the x_0-head is too close to ground truth and
cold-diffusion is degenerate.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff
from subdiff.data import build_eval_dataloader


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--prompt_ratio', type=float, default=0.25)
    p.add_argument('--num_samples', type=int, default=512,
                   help='val images to evaluate over')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ts', type=str, default='1.0,0.75,0.5,0.25,0.0',
                   help='comma-separated t values to measure at')
    return p.parse_args()


def _build_model(cfg, device):
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
    return SubDiff(
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
        dual_decoder=cfg['model'].get('dual_decoder', False),
        clean_ratio=cfg['model'].get('clean_ratio', 0.25),
        pixel_loss_weight=cfg['model'].get('pixel_loss_weight', 1.0),
        qk_norm=cfg['model'].get('qk_norm', False),
        dit_minimal_head=cfg['model'].get('dit_minimal_head', False),
        use_indicators=cfg['model'].get('use_indicators', False),
        pos_embed_type=cfg['model'].get('pos_embed_type', 'sincos'),
        flow_matching=cfg['diffusion'].get('flow_matching', False),
        rf_t_sampling=cfg['diffusion'].get('rf_t_sampling', 'logit_normal'),
    ).to(device)


@torch.no_grad()
def main():
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = _build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model']
    if getattr(model.encoder, 'pos_embed_type', 'learnable') == 'sincos':
        state = {k: v for k, v in state.items() if not k.endswith('encoder.pos_embed')}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded ep={ckpt['epoch']} avg_loss={ckpt.get('avg_loss', 'n/a')}")

    val_loader, _ = build_eval_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=args.batch_size, num_workers=4, backend='torch',
    )

    ts = [float(x) for x in args.ts.split(',')]
    N_target = args.num_samples
    img_size = int(model.patch_size * (model.num_patches ** 0.5))

    # accumulators per t
    stats = {t: {'mse_mu_x0': [], 'mse_zero_x0': [], 'n': 0} for t in ts}

    seen = 0
    for imgs, _ in val_loader:
        if seen >= N_target:
            break
        imgs = imgs.to(device)
        B = imgs.shape[0]
        N = model.num_patches
        target_patches = model.patchify(imgs)              # (B, N, D)

        # 25% clean prompt mask (same per t for fair comparison)
        rand = torch.rand(B, N, device=device)
        k = int(round(N * args.prompt_ratio))
        ids_keep = torch.argsort(rand, dim=1)[:, :k]
        prompt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        prompt_mask.scatter_(1, ids_keep, True)
        noisy_mask = ~prompt_mask                          # True = noisy
        m = noisy_mask.unsqueeze(-1).float()               # (B, N, 1)

        # Same epsilon across t for paired comparison
        eps = torch.randn_like(target_patches)

        for t_val in ts:
            t_b = torch.full((B, 1, 1), t_val, device=device)
            x_t = (1 - t_b) * target_patches + t_b * eps
            mixed_patches = (1 - m) * target_patches + m * x_t
            mixed_imgs = model.unpatchify(mixed_patches, img_size=img_size)

            t_cont = torch.full((B,), t_val, device=device)
            t_int = (t_cont * (model.diffusion.num_timesteps - 1)).long()
            cls_token, patch_tokens = model._encode_with_indicators(
                mixed_imgs, noisy_mask, t_int)

            pred_x0 = model.decoder_pix(patch_tokens)
            pred_x0 = model._apply_conv_refine(pred_x0)

            # MSE per noisy patch (avg over patch_dim), then collect
            sq_err = ((pred_x0 - target_patches) ** 2).mean(dim=-1)   # (B, N)
            sq_x0  = (target_patches ** 2).mean(dim=-1)               # (B, N)

            mse_mu = sq_err[noisy_mask].cpu().numpy()
            mse_zero = sq_x0[noisy_mask].cpu().numpy()

            stats[t_val]['mse_mu_x0'].append(mse_mu)
            stats[t_val]['mse_zero_x0'].append(mse_zero)
            stats[t_val]['n'] += int(noisy_mask.sum().item())

        seen += B
        print(f"  processed {seen}/{N_target} images")

    print()
    print("=" * 78)
    print(f"Run X x_0-head distance to true x_0 on noisy positions")
    print(f"  ckpt: {args.checkpoint}")
    print(f"  prompt_ratio: {args.prompt_ratio}  (={k}/{N} clean tokens)")
    print(f"  evaluated over {seen} images, {stats[ts[0]]['n']} noisy patches per t")
    print("=" * 78)
    print(f"{'t':>5}  {'MSE(μ, x_0)':>14}  {'MSE(0, x_0)':>14}  {'frac_var_unexpl':>17}  {'p50':>8}  {'p90':>8}")
    print("-" * 78)
    for t_val in ts:
        mu = np.concatenate(stats[t_val]['mse_mu_x0'])
        zero = np.concatenate(stats[t_val]['mse_zero_x0'])
        mean_mu = float(mu.mean())
        mean_zero = float(zero.mean())
        frac = mean_mu / mean_zero
        p50 = float(np.median(mu))
        p90 = float(np.percentile(mu, 90))
        print(f"{t_val:>5.2f}  {mean_mu:>14.5f}  {mean_zero:>14.5f}  {frac:>17.4f}  {p50:>8.5f}  {p90:>8.5f}")
    print("=" * 78)
    print()
    print("Interpretation:")
    print("  MSE(μ, x_0)  = average per-pixel MSE of x_0-head pred vs ground truth")
    print("  MSE(0, x_0)  = baseline if predicting zero (≈ 1.0 since normalized)")
    print("  frac_var_unexpl = MSE(μ, x_0) / MSE(0, x_0) — fraction of variance left to explain")
    print()
    print("Decision rule for Run Y (cold-RF) refiner:")
    print("  - If MSE(μ, x_0) at t=1.0 > 0.05 → strong residual, refiner has work")
    print("  - If 0.01 < MSE(μ, x_0) at t=1.0 < 0.05 → some residual, marginal")
    print("  - If MSE(μ, x_0) at t=1.0 < 0.01 → x_0-head too good, refiner degenerate")


if __name__ == '__main__':
    main()
