"""
Sample from a flow-matching (Rectified Flow) checkpoint using Euler ODE solver.

Usage:
  python scripts/sample_flow.py \
      --config configs/pretrain_vit_b16_naive_rf.yaml \
      --checkpoint logs_naive_rf/checkpoints/checkpoint_best.pth \
      --num_samples 16 --num_steps 50 \
      --output_dir samples_rf/
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--num_samples', type=int, default=16)
    p.add_argument('--num_steps', type=int, default=50,
                   help='ODE integration steps (Euler; RF usually needs 20-50)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--output_dir', type=str, default='samples_rf/')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--solver', type=str, default='euler', choices=['euler', 'heun'])
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
        naive_ddpm=cfg['model'].get('naive_ddpm', False),
        qk_norm=cfg['model'].get('qk_norm', False),
        dit_minimal_head=cfg['model'].get('dit_minimal_head', False),
        use_conv_refine=cfg['model'].get('use_conv_refine', False),
        pos_embed_type=cfg['model'].get('pos_embed_type', 'sincos'),
        flow_matching=cfg.get('diffusion', {}).get('flow_matching', False),
        rf_t_sampling=cfg.get('diffusion', {}).get('rf_t_sampling', 'logit_normal'),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model']
    if getattr(model.encoder, 'pos_embed_type', 'learnable') == 'sincos':
        state = {k: v for k, v in state.items() if not k.endswith('encoder.pos_embed')}
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"avg_loss={ckpt.get('avg_loss', 'n/a')}")
    model.eval()
    return model


def _predict_velocity(model, x, t_scalar):
    """Compute v̂ = model(x, t) for RF. t_scalar ∈ (0, 1), broadcast to (B,)."""
    img_size = x.shape[-1]
    B = x.shape[0]
    t_cont = torch.full((B,), t_scalar, device=x.device)
    t_int = (t_cont * (model.diffusion.num_timesteps - 1)).long()
    cls_token, patch_tokens = model._encode_with_time(x, t_int)
    pred_v_patches = model.decoder(patch_tokens)
    pred_v_patches = model._apply_conv_refine(pred_v_patches)
    return model.unpatchify(pred_v_patches, img_size=img_size)


@torch.no_grad()
def rf_sample(model, batch_size, img_size, num_steps, device, solver='euler'):
    """Rectified Flow sampling: solve dx/dt = -v(x, t) from t=1 (noise) to t=0 (data).

    Note: in RF we parameterize x_t = (1-t)*x_0 + t*eps, so dx_t/dt = eps - x_0 = v.
    Going BACKWARDS from noise (t=1) to data (t=0), we integrate -v.
    """
    # Initial state at t=1 is pure noise
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = ts[1:] - ts[:-1]  # negative values (going from 1 → 0)

    for i, (t_cur, step) in enumerate(zip(ts[:-1], dt)):
        v1 = _predict_velocity(model, x, t_cur.item())
        if solver == 'heun' and i < num_steps - 1:
            # Predictor-corrector: trial step, re-evaluate v at t_next, average
            x_trial = x + step * v1
            t_next = ts[i + 1]
            v2 = _predict_velocity(model, x_trial, t_next.item())
            x = x + step * 0.5 * (v1 + v2)
        else:
            x = x + step * v1

        if (i + 1) % max(1, num_steps // 10) == 0 or i == num_steps - 1:
            print(f"  step {i+1}/{num_steps}  t={t_cur.item():.3f}  "
                  f"x range=[{x.min().item():.2f}, {x.max().item():.2f}]")

    return x.clamp(-3, 3)


def denormalize(imgs):
    return (imgs * STD.to(imgs.device) + MEAN.to(imgs.device)).clamp(0, 1)


def save_grid(imgs, path, nrow=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = imgs.shape[0]
    ncol = nrow
    nrow_a = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_a, ncol, figsize=(3 * ncol, 3 * nrow_a))
    if nrow_a == 1:
        axes = axes[None, :]
    for i in range(nrow_a * ncol):
        ax = axes[i // ncol, i % ncol]
        if i < n:
            img = imgs[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    assert cfg.get('diffusion', {}).get('flow_matching', False), \
        "sample_flow.py requires config with diffusion.flow_matching: true"

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(cfg, args.checkpoint, device)
    img_size = cfg['data']['image_size']
    os.makedirs(args.output_dir, exist_ok=True)

    all_samples = []
    n_done = 0
    while n_done < args.num_samples:
        bs = min(args.batch_size, args.num_samples - n_done)
        print(f"\nBatch {n_done}/{args.num_samples} (bs={bs}, solver={args.solver})...")
        x = rf_sample(model, bs, img_size, args.num_steps, device, solver=args.solver)
        all_samples.append(denormalize(x))
        n_done += bs

    samples = torch.cat(all_samples, dim=0)
    grid_path = os.path.join(args.output_dir, 'grid.png')
    save_grid(samples, grid_path, nrow=4)
    print(f"\nSaved grid: {grid_path}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i, img in enumerate(samples):
        path = os.path.join(args.output_dir, f'sample_{i:03d}.png')
        plt.imsave(path, img.cpu().permute(1, 2, 0).numpy())


if __name__ == '__main__':
    main()
