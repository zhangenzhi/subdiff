"""
Inpainting with a Run 10-style RF+MAE checkpoint.

Run 10's masked-token branch trained the encoder/decoder to predict v at
masked positions from the surrounding visible context. At masked positions,
the optimal predictor is pred_v* = -x̂_0 (because ε is independent of
context there). So we recover x̂_0 simply by negating pred_v at the masked
positions and composing with the user-provided clean patches at the rest.

This is single-pass MAE, not iterative diffusion (RePaint). It exercises
exactly the inductive bias that Run 10 was trained for.

Usage:
  python scripts/inpaint_rf_mae.py \
    --config configs/pretrain_vit_b16_naive_rf_mae.yaml \
    --checkpoint logs_naive_rf_mae/checkpoints/checkpoint_best.pth \
    --mask_ratio 0.5 --num_samples 4 --t_val 0.5 \
    --output_dir samples_rf_mae_inpaint/
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
    p.add_argument('--mask_ratio', type=float, default=0.5,
                   help='Fraction of patches to mask out (be filled).')
    p.add_argument('--t_val', type=float, default=0.5,
                   help='RF time used in the model forward. 0.5 is the '
                        'logit-normal peak from training.')
    p.add_argument('--num_samples', type=int, default=4)
    p.add_argument('--output_dir', type=str, default='samples_rf_mae_inpaint/')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def _build_model(cfg, device):
    """Mirrors visualize._build_model_from_cfg without depending on it."""
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
        naive_ddpm=cfg.get('model', {}).get('naive_ddpm', False),
        qk_norm=cfg.get('model', {}).get('qk_norm', False),
        dit_minimal_head=cfg.get('model', {}).get('dit_minimal_head', False),
        pos_embed_type=cfg.get('model', {}).get('pos_embed_type', 'sincos'),
        flow_matching=cfg.get('diffusion', {}).get('flow_matching', False),
        rf_t_sampling=cfg.get('diffusion', {}).get('rf_t_sampling', 'logit_normal'),
        rf_mae_enabled=cfg.get('diffusion', {}).get('rf_mae_enabled', False),
        rf_mae_max_mask=cfg.get('diffusion', {}).get('rf_mae_max_mask', 0.5),
        mae_aux_weight=cfg.get('diffusion', {}).get('mae_aux_weight', 0.1),
    ).to(device)


@torch.no_grad()
def inpaint_one_pass(model, imgs, patch_mask, t_val=0.5):
    """Single-pass MAE-style inpainting using Run 10's masked branch.

    Args:
        imgs: (B, 3, H, W) clean ImageNet-normalized images
        patch_mask: (B, N) bool, True = position to be filled
        t_val: scalar t passed to the model (default 0.5)
    Returns:
        composite imgs (B, 3, H, W): clean known patches + predicted unknown
        pred_x0_at_masked imgs: just the model's prediction at masked positions
            (gray elsewhere) for visualization
    """
    assert getattr(model, 'rf_mae_enabled', False), \
        "Checkpoint must come from RF+MAE training (rf_mae_enabled=True)."

    device = imgs.device
    B = imgs.shape[0]
    N = model.num_patches
    img_size = int(model.patch_size * (N ** 0.5))

    target_patches = model.patchify(imgs)                 # (B, N, D)

    # Noise the underlying clean image at level t_val (encoder is most
    # in-distribution at t≈0.5 from the logit-normal training)
    t_cont = torch.full((B,), t_val, device=device)
    t_b = t_cont.view(B, 1, 1)
    eps = torch.randn_like(target_patches)
    x_t_patches = (1 - t_b) * target_patches + t_b * eps
    x_t_imgs = model.unpatchify(x_t_patches, img_size=img_size)

    # patch_embed all positions, then replace unknown ones with mask_token
    patch_tokens = model.encoder.patch_embed(x_t_imgs)    # (B, N, D)
    m = patch_mask.unsqueeze(-1).float()                  # (B, N, 1) — 1 = unknown
    mask_token = model.rf_mask_token.expand(B, N, -1)
    patch_tokens = (1 - m) * patch_tokens + m * mask_token

    # Encoder (DiT adaLN-Zero) with time conditioning
    t_int = (t_cont * (model.diffusion.num_timesteps - 1)).long()
    c = model.time_embed(t_int)
    cls, enc_tokens = model.encoder.forward_patches(patch_tokens, c)

    pred_v = model.decoder(enc_tokens)                    # (B, N, patch_dim)
    pred_v = model._apply_conv_refine(pred_v)

    # x̂_0 ≈ -pred_v at masked positions (training-optimal predictor; see docstring)
    pred_x0_at_masked = -pred_v

    # Composite: known = original x_0, unknown = predicted x̂_0
    composite_patches = (1 - m) * target_patches + m * pred_x0_at_masked

    # "Filled-only" view: gray where known, prediction where unknown
    gray = torch.zeros_like(target_patches)               # gray after denormalize
    filled_only = (1 - m) * gray + m * pred_x0_at_masked

    # "Masked input" view: gray where unknown, original where known
    masked_input = (1 - m) * target_patches + m * gray

    return (model.unpatchify(composite_patches, img_size=img_size),
            model.unpatchify(filled_only, img_size=img_size),
            model.unpatchify(masked_input, img_size=img_size))


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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    B, _, H, W = imgs.shape
    N = model.num_patches

    # Random patch mask, same per sample for consistency
    rand = torch.rand(B, N, device=device)
    k = int(round(N * args.mask_ratio))
    ids_shuffle = torch.argsort(rand, dim=1)
    ids_mask = ids_shuffle[:, :k]
    patch_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    patch_mask.scatter_(1, ids_mask, True)

    composite, filled, masked_input = inpaint_one_pass(
        model, imgs, patch_mask, t_val=args.t_val
    )

    titles = [
        "Original x_0",
        f"Masked input\n(mask_ratio={args.mask_ratio:.2f})",
        f"Inpaint composite\n(t={args.t_val})",
        "Predicted patches only",
    ]
    save_grid(
        [denormalize(imgs), denormalize(masked_input),
         denormalize(composite), denormalize(filled)],
        titles,
        os.path.join(args.output_dir, f"inpaint_t{args.t_val}_r{args.mask_ratio}.png"),
        nrow=B,
    )


if __name__ == '__main__':
    main()
