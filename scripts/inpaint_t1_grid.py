"""
4-column visualization at t=1.0 (pure noise on noisy patches).

Shows: Original | Clean prompt (gray elsewhere) | Recon (gray elsewhere) | Composite

This is the maximum-noise case for the dual-RF inpainting model — the model
must reconstruct the noisy 75% of patches purely from the 25% clean prompt
context (no information left in x_t at noisy positions when t=1).
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
    p.add_argument('--prompt_ratio', type=float, default=0.25)
    p.add_argument('--num_samples', type=int, default=4)
    p.add_argument('--output_dir', type=str, default='samples_rf_dual_inpaint_t1/')
    p.add_argument('--seed', type=int, default=42)
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


def denormalize(imgs):
    return (imgs * STD.to(imgs.device) + MEAN.to(imgs.device)).clamp(0, 1)


def save_grid(images_list, titles, save_path, nrow):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncol = len(images_list)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    if nrow == 1:
        axes = axes[None, :]
    for r in range(nrow):
        for c in range(ncol):
            img = images_list[c][r].cpu().permute(1, 2, 0).numpy()
            axes[r, c].imshow(np.clip(img, 0, 1))
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


@torch.no_grad()
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
    print(f"Loaded ep={ckpt['epoch']} avg_loss={ckpt.get('avg_loss', 'n/a')}")

    val_loader, _ = build_eval_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=args.num_samples, num_workers=2, backend='torch',
    )
    imgs, _ = next(iter(val_loader))
    imgs = imgs.to(device)
    B = imgs.shape[0]
    N = model.num_patches
    img_size = int(model.patch_size * (N ** 0.5))

    target_patches = model.patchify(imgs)

    # Build prompt mask: 25% clean
    rand = torch.rand(B, N, device=device)
    k = int(round(N * args.prompt_ratio))
    ids_keep = torch.argsort(rand, dim=1)[:, :k]
    prompt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    prompt_mask.scatter_(1, ids_keep, True)
    noisy_mask = ~prompt_mask                          # True = noisy (to fill)
    m = noisy_mask.unsqueeze(-1).float()               # (B, N, 1)

    # t = 1.0 means x_t = ε at noisy positions (pure noise, no x_0 signal)
    eps = torch.randn_like(target_patches)
    mixed_patches = (1 - m) * target_patches + m * eps   # clean prompt + pure noise
    mixed_imgs = model.unpatchify(mixed_patches, img_size=img_size)

    # Time embed: integer index for sinusoidal
    t_cont = torch.full((B,), 1.0, device=device)
    t_int = (t_cont * (model.diffusion.num_timesteps - 1)).long()  # = num_timesteps-1
    cls_token, patch_tokens = model._encode_with_indicators(mixed_imgs, noisy_mask, t_int)

    pred_x0 = model.decoder_pix(patch_tokens)
    pred_x0 = model._apply_conv_refine(pred_x0)

    # ---- 4 viz columns -----------------------------------------------------
    # 1. Original x_0
    orig = denormalize(imgs)

    # 2. Clean prompt — real x_0 at prompt positions, gray elsewhere
    gray_patches = torch.zeros_like(target_patches)     # zero = gray after denorm
    clean_only_patches = (1 - m) * target_patches + m * gray_patches
    clean_only = denormalize(model.unpatchify(clean_only_patches, img_size=img_size))

    # 3. Reconstruction — model's pred_x0 at noisy positions, gray elsewhere
    recon_only_patches = (1 - m) * gray_patches + m * pred_x0
    recon_only = denormalize(model.unpatchify(recon_only_patches, img_size=img_size))

    # 4. Composite — clean prompt + recon
    composite_patches = (1 - m) * target_patches + m * pred_x0
    composite = denormalize(model.unpatchify(composite_patches, img_size=img_size))

    save_grid(
        [orig, clean_only, recon_only, composite],
        ["Original x_0",
         f"Clean prompt only\n({args.prompt_ratio:.0%} kept, gray=noise)",
         "Recon only\n(model fill, gray=prompt)",
         "Composite\n(prompt + recon)"],
        os.path.join(args.output_dir, f"t1.0_p{args.prompt_ratio}.png"),
        nrow=B,
    )


if __name__ == '__main__':
    main()
