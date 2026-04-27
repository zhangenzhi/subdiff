"""
Dual-RF inpainting (Run 11) — sweep over prompt ratio.

Run 11 was trained with clean_ratio = 0.25 (25% of patches are clean prompt).
This script tests how the model behaves when the prompt ratio at inference
diverges from training:
  - 0.25 = in-distribution (matches training)
  - 0.15, 0.05 = lower than trained → growing OOD
  - 0.00 = no prompt at all (fully OOD; analogous to Run 7's mode-collapse case)

For each prompt ratio r:
  - Pick r·N patches randomly per sample as clean prompt
  - The other (1-r)·N patches get RF noise at t=0.5
  - Forward through the model; use x_0-head's prediction at noisy positions
  - Composite: real x_0 at prompt positions, predicted x_0 at noisy

Usage:
  python scripts/inpaint_rf_dual.py \
    --config configs/pretrain_vit_b16_dual_rf.yaml \
    --checkpoint logs_dual_rf/checkpoints/checkpoint_latest.pth \
    --prompt_ratios 0.25 0.15 0.05 0.0 --t_val 0.5 --num_samples 4 \
    --output_dir samples_rf_dual_inpaint/
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

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--prompt_ratios', type=float, nargs='+',
                   default=[0.25, 0.15, 0.05, 0.0],
                   help='Prompt fractions to evaluate (training used 0.25). '
                        'If a single value is given, used for all columns.')
    p.add_argument('--t_vals', type=float, nargs='+',
                   default=None,
                   help='RF time(s) used per column. If given, must align with '
                        '--prompt_ratios (same length OR one of them is len 1).')
    p.add_argument('--t_val', type=float, default=0.5,
                   help='Default RF time when --t_vals is not given.')
    p.add_argument('--num_samples', type=int, default=4)
    p.add_argument('--output_dir', type=str, default='samples_rf_dual_inpaint/')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def _build_model(cfg, device):
    """Build SubDiff with all flags wired through (same as training)."""
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
        dual_decoder=cfg.get('model', {}).get('dual_decoder', False),
        clean_ratio=cfg.get('model', {}).get('clean_ratio', 0.25),
        pixel_loss_weight=cfg.get('model', {}).get('pixel_loss_weight', 1.0),
        qk_norm=cfg.get('model', {}).get('qk_norm', False),
        dit_minimal_head=cfg.get('model', {}).get('dit_minimal_head', False),
        use_indicators=cfg.get('model', {}).get('use_indicators', False),
        pos_embed_type=cfg.get('model', {}).get('pos_embed_type', 'sincos'),
        flow_matching=cfg.get('diffusion', {}).get('flow_matching', False),
        rf_t_sampling=cfg.get('diffusion', {}).get('rf_t_sampling', 'logit_normal'),
    ).to(device)


@torch.no_grad()
def inpaint_one_pass_dual(model, imgs, prompt_mask, t_val=0.5):
    """Single forward pass through the dual-RF model.

    Args:
        imgs: (B, 3, H, W) clean ImageNet-normalized images
        prompt_mask: (B, N) bool — True = clean prompt position
        t_val: RF time scalar
    Returns:
        composite: clean at prompt + x_0-head fill at noisy (B,3,H,W)
        v_x0: v-head's Euler→0 prediction (B,3,H,W) for comparison
        mixed_input: the actual encoder input (B,3,H,W) for visualization
    """
    device = imgs.device
    B = imgs.shape[0]
    N = model.num_patches
    img_size = int(model.patch_size * (N ** 0.5))

    target_patches = model.patchify(imgs)
    t_cont = torch.full((B,), t_val, device=device)
    t_b = t_cont.view(B, 1, 1)

    eps = torch.randn_like(target_patches)
    x_t = (1 - t_b) * target_patches + t_b * eps

    noisy_mask = ~prompt_mask                   # (B, N) — True = noisy
    m = noisy_mask.unsqueeze(-1).float()        # (B, N, 1)
    mixed_patches = (1 - m) * target_patches + m * x_t
    mixed_imgs = model.unpatchify(mixed_patches, img_size=img_size)

    t_int = (t_cont * (model.diffusion.num_timesteps - 1)).long()
    cls_token, patch_tokens = model._encode_with_indicators(mixed_imgs, noisy_mask, t_int)

    pred_v = model.decoder(patch_tokens)
    pred_x0 = model.decoder_pix(patch_tokens)
    pred_v = model._apply_conv_refine(pred_v)
    pred_x0 = model._apply_conv_refine(pred_x0)

    # v-head Euler→0 (over noisy positions only; clean stays)
    v_x0_patches = mixed_patches - t_b * pred_v
    v_x0_patches = (1 - m) * target_patches + m * v_x0_patches

    # x_0-head composite: clean prompt + x_0-head prediction on noisy
    composite_patches = (1 - m) * target_patches + m * pred_x0

    return (model.unpatchify(composite_patches, img_size=img_size),
            model.unpatchify(v_x0_patches, img_size=img_size),
            mixed_imgs)


def denormalize(imgs):
    return (imgs * STD.to(imgs.device) + MEAN.to(imgs.device)).clamp(0, 1)


def save_grid(images_list, titles, save_path, nrow):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncol = len(images_list)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow))
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


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    model = _build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model']
    if getattr(model.encoder, 'pos_embed_type', 'learnable') == 'sincos':
        state = {k: v for k, v in state.items() if not k.endswith('encoder.pos_embed')}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt['epoch']} avg_loss={ckpt.get('avg_loss', 'n/a')}")

    val_loader, _ = build_eval_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=args.num_samples, num_workers=2, backend='torch',
    )
    imgs, _ = next(iter(val_loader))
    imgs = imgs.to(device)
    B = imgs.shape[0]
    N = model.num_patches

    # Build per-column (prompt_ratio, t_val) pairs
    if args.t_vals is None:
        pairs = [(r, args.t_val) for r in args.prompt_ratios]
        out_tag = f"prompt_sweep_t{args.t_val}"
    elif len(args.t_vals) == len(args.prompt_ratios):
        pairs = list(zip(args.prompt_ratios, args.t_vals))
        out_tag = "rt_sweep"
    elif len(args.prompt_ratios) == 1:
        pairs = [(args.prompt_ratios[0], t) for t in args.t_vals]
        out_tag = f"t_sweep_p{args.prompt_ratios[0]}"
    elif len(args.t_vals) == 1:
        pairs = [(r, args.t_vals[0]) for r in args.prompt_ratios]
        out_tag = f"prompt_sweep_t{args.t_vals[0]}"
    else:
        raise ValueError("Mismatched lengths for --prompt_ratios and --t_vals")

    cols = [denormalize(imgs)]
    titles = ["Original"]

    for r, tv in pairs:
        torch.manual_seed(args.seed + int(r * 1000) + int(tv * 100))
        rand = torch.rand(B, N, device=device)
        k = int(round(N * r))
        if k > 0:
            ids_shuffle = torch.argsort(rand, dim=1)
            ids_keep = ids_shuffle[:, :k]
            prompt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            prompt_mask.scatter_(1, ids_keep, True)
        else:
            prompt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        composite, v_x0, mixed_in = inpaint_one_pass_dual(
            model, imgs, prompt_mask, t_val=tv
        )
        cols.append(denormalize(composite))
        titles.append(f"r={r:.2f}, t={tv:.2f}")

    save_grid(
        cols, titles,
        os.path.join(args.output_dir, f"{out_tag}.png"),
        nrow=B,
    )


if __name__ == '__main__':
    main()
