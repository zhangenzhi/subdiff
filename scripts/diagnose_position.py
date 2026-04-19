"""Diagnostic: check if encoder produces position-dependent output.

For a given checkpoint, feed pure noise and check:
1. pos_embed norm (sanity check)
2. How different patch token outputs are across positions
3. How different decoder outputs are across positions

If all positions produce near-identical outputs → model ignores position.
"""
import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subdiff.model import SubDiff


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--t', type=int, default=500)
    return p.parse_args()


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}, avg_loss={ckpt.get('avg_loss', 'n/a')}\n")

    # 1. Check pos_embed
    pos = model.encoder.pos_embed  # (1, N+1, D) or (1, N, D)
    print(f"pos_embed shape: {pos.shape}")
    print(f"pos_embed norm overall: {pos.norm().item():.4f}")
    print(f"pos_embed std: {pos.std().item():.4f}")
    print(f"pos_embed range: [{pos.min().item():.4f}, {pos.max().item():.4f}]")

    # Check that different positions have different pos_embed
    patch_pos = pos[:, 1:, :] if pos.shape[1] == 197 else pos  # strip cls if present
    diffs = []
    for i in [0, 50, 100, 150]:
        for j in [0, 50, 100, 150]:
            if i < j:
                d = (patch_pos[0, i] - patch_pos[0, j]).norm().item()
                diffs.append(d)
                print(f"  ||pos[{i}] - pos[{j}]||: {d:.4f}")
    print(f"avg pos_embed pair difference: {sum(diffs)/len(diffs):.4f}")
    print()

    # 2. Feed pure noise and check encoder output diversity
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224, device=device)
    t = torch.tensor([args.t], device=device, dtype=torch.long)

    with torch.no_grad():
        if model.naive_ddpm:
            cls_tok, patch_tokens = model._encode_with_time(x, t)
        else:
            noisy_mask = torch.ones(1, model.num_patches, dtype=torch.bool, device=device)
            cls_tok, patch_tokens = model._encode_with_indicators(x, noisy_mask, t)

    # patch_tokens: (1, 196, D)
    print(f"patch_tokens shape: {patch_tokens.shape}")
    print(f"patch_tokens std across positions (per-dim mean): {patch_tokens[0].std(dim=0).mean().item():.4f}")
    print(f"patch_tokens std across feature dim (per-position mean): {patch_tokens[0].std(dim=1).mean().item():.4f}")

    # Compare specific positions
    print("\nToken pair differences (||token_i - token_j||):")
    for i in [0, 50, 100, 150, 195]:
        for j in [0, 50, 100, 150, 195]:
            if i < j:
                d = (patch_tokens[0, i] - patch_tokens[0, j]).norm().item()
                print(f"  ||tok[{i}] - tok[{j}]||: {d:.4f}")
    print()

    # 3. Check decoder output diversity
    with torch.no_grad():
        pred = model.decoder(patch_tokens)  # (1, 196, patch_dim)
    print(f"decoder output shape: {pred.shape}")
    print(f"decoder output std across positions (per-dim mean): {pred[0].std(dim=0).mean().item():.4f}")

    print("\nDecoder output pair differences:")
    for i in [0, 50, 100, 150, 195]:
        for j in [0, 50, 100, 150, 195]:
            if i < j:
                d = (pred[0, i] - pred[0, j]).norm().item()
                mean_mag = (pred[0, i].abs().mean() + pred[0, j].abs().mean()).item() / 2
                ratio = d / (mean_mag * pred.shape[-1] ** 0.5 + 1e-6)
                print(f"  ||pred[{i}] - pred[{j}]||: {d:.4f}, relative: {ratio:.4f}")


if __name__ == '__main__':
    main()
