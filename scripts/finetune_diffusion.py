"""
Diffusion finetuning: train a full diffusion model (UNet decoder) using
the SubDiff pretrained encoder as the conditioning backbone.

Supports switching to smaller patch sizes for fine-grained generation.

Usage:
  Single GPU:
    python scripts/finetune_diffusion.py --config configs/finetune_diffusion.yaml \
        --checkpoint logs/checkpoints/checkpoint_final.pth

  Multi-GPU:
    torchrun --nproc_per_node=8 scripts/finetune_diffusion.py \
        --config configs/finetune_diffusion.yaml \
        --checkpoint logs/checkpoints/checkpoint_final.pth
"""

import os
import sys
import argparse
import math
import yaml

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff
from subdiff.diffusion import PatchDiffusion
from subdiff.data import build_pretrain_dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help='pretrained checkpoint (omit for scratch baseline)')
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cosine_lr_schedule(optimizer, epoch, total_epochs, warmup_epochs, lr, min_lr):
    if epoch < warmup_epochs:
        cur_lr = lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cur_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class DiffusionDecoderBlock(nn.Module):
    """Transformer block with cross-attention to encoder features + time conditioning."""

    def __init__(self, dim, num_heads, encoder_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.norm_enc = nn.LayerNorm(encoder_dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, kdim=encoder_dim, vdim=encoder_dim
        )

        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        # Time conditioning via adaptive layer norm (simplified)
        self.time_proj = nn.Linear(dim, dim * 2)

    def forward(self, x, encoder_feat, time_emb):
        # Time modulation
        scale, shift = self.time_proj(time_emb).unsqueeze(1).chunk(2, dim=-1)

        # Self-attention
        normed = self.norm1(x)
        normed = normed * (1 + scale) + shift
        x = x + self.self_attn(normed, normed, normed, need_weights=False)[0]

        # Cross-attention to encoder features
        x = x + self.cross_attn(
            self.norm2(x), self.norm_enc(encoder_feat), encoder_feat,
            need_weights=False
        )[0]

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class DiffusionDecoder(nn.Module):
    """
    Transformer-based diffusion decoder.

    Takes noisy patch tokens + encoder features + timestep,
    predicts noise (epsilon) for each patch.
    """

    def __init__(self, patch_size=8, num_patches=784, in_dim=None,
                 decoder_dim=512, depth=6, num_heads=8, encoder_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        patch_dim = patch_size ** 2 * 3
        if in_dim is None:
            in_dim = patch_dim

        self.input_proj = nn.Linear(in_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.time_embed = TimeEmbedding(decoder_dim)

        self.blocks = nn.ModuleList([
            DiffusionDecoderBlock(decoder_dim, num_heads, encoder_dim)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(decoder_dim)
        self.output_proj = nn.Linear(decoder_dim, patch_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, noisy_patches, encoder_feat, t):
        """
        Args:
            noisy_patches: (B, N, patch_dim) noisy patch pixels
            encoder_feat: (B, N_enc, encoder_dim) encoder patch tokens
            t: (B,) diffusion timestep

        Returns:
            pred_noise: (B, N, patch_dim) predicted noise
        """
        x = self.input_proj(noisy_patches) + self.pos_embed
        time_emb = self.time_embed(t)

        for block in self.blocks:
            x = block(x, encoder_feat, time_emb)

        x = self.norm(x)
        return self.output_proj(x)


class DiffusionFinetune(nn.Module):
    """
    Full diffusion model using pretrained SubDiff encoder.

    The encoder processes clean (or partially noisy) images to produce
    conditioning features. The diffusion decoder learns to denoise at
    all timesteps, conditioned on encoder features.

    Supports a different (smaller) patch size than the encoder.
    """

    def __init__(self, encoder, encoder_dim=768, encoder_patch_size=16,
                 decoder_patch_size=8, img_size=224,
                 decoder_dim=512, decoder_depth=6, decoder_num_heads=8,
                 num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 schedule_type='linear', freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.encoder_patch_size = encoder_patch_size
        self.decoder_patch_size = decoder_patch_size
        self.img_size = img_size

        self.num_decoder_patches = (img_size // decoder_patch_size) ** 2
        decoder_patch_dim = decoder_patch_size ** 2 * 3

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = DiffusionDecoder(
            patch_size=decoder_patch_size,
            num_patches=self.num_decoder_patches,
            in_dim=decoder_patch_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            encoder_dim=encoder_dim,
        )

        self.diffusion = PatchDiffusion(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type,
        )

        self.num_timesteps = num_timesteps

    def patchify(self, imgs, patch_size):
        p = patch_size
        return rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)

    def unpatchify(self, patches, patch_size):
        p = patch_size
        h = w = self.img_size // p
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=h, w=w, p1=p, p2=p, c=3)

    def forward(self, imgs):
        """
        Args:
            imgs: (B, C, H, W) clean images

        Returns:
            loss: diffusion training loss (predict noise)
            log_dict: metrics
        """
        B = imgs.shape[0]
        device = imgs.device

        # Encoder conditioning (on clean images)
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            _, encoder_feat = self.encoder(imgs)  # (B, N_enc, D)

        # Patchify at decoder resolution
        target_patches = self.patchify(imgs, self.decoder_patch_size)  # (B, N_dec, patch_dim)

        # Sample random timesteps (full range for standard diffusion training)
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffusion
        noisy_patches, noise = self.diffusion.add_noise(target_patches, t)

        # Predict noise
        pred_noise = self.decoder(noisy_patches, encoder_feat, t)

        # Simple MSE loss on noise prediction
        loss = nn.functional.mse_loss(pred_noise, noise)

        log_dict = {
            'loss': loss.item(),
            't_mean': t.float().mean().item(),
        }
        return loss, log_dict

    @torch.no_grad()
    def sample(self, imgs_cond, num_steps=None):
        """
        Generate images via DDPM sampling, conditioned on encoder features.

        Args:
            imgs_cond: (B, C, H, W) conditioning images for the encoder
            num_steps: number of denoising steps (default: num_timesteps)

        Returns:
            generated: (B, C, H, W) generated images
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        B = imgs_cond.shape[0]
        device = imgs_cond.device

        # Get encoder features
        _, encoder_feat = self.encoder(imgs_cond)

        # Start from pure noise
        patch_dim = self.decoder_patch_size ** 2 * 3
        x = torch.randn(B, self.num_decoder_patches, patch_dim, device=device)

        # DDPM reverse process
        for i in reversed(range(num_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            pred_noise = self.decoder(x, encoder_feat, t)

            alpha = 1 - self.diffusion.betas[i]
            alpha_cumprod = self.diffusion.alphas_cumprod[i]
            alpha_cumprod_prev = self.diffusion.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0)

            # DDPM update
            pred_x0 = (x - (1 - alpha_cumprod).sqrt() * pred_noise) / alpha_cumprod.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3, 3)  # stability clamp

            # Posterior mean
            posterior_mean = (
                alpha_cumprod_prev.sqrt() * self.diffusion.betas[i] / (1 - alpha_cumprod) * pred_x0
                + alpha.sqrt() * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * x
            )

            if i > 0:
                noise = torch.randn_like(x)
                posterior_var = self.diffusion.betas[i] * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                x = posterior_mean + posterior_var.sqrt() * noise
            else:
                x = posterior_mean

        return self.unpatchify(x, self.decoder_patch_size)


def _transfer_decoder_weights(pretrained_decoder, diff_decoder, verbose=False):
    """Transfer compatible weights from pretrained decoder to diffusion decoder.

    Mapping (pretrained Block → DiffusionDecoderBlock):
      attn.qkv.weight/bias     → self_attn.in_proj_weight/bias
      attn.proj.weight/bias     → self_attn.out_proj.weight/bias
      norm1                     → norm1
      norm2 (pre-MLP)           → norm3 (pre-MLP in diffusion block)
      mlp.fc1.weight/bias       → mlp.0.weight/bias
      mlp.fc2.weight/bias       → mlp.2.weight/bias

    Also transfers:
      proj (encoder_dim→decoder_dim) → input_proj (if shapes match)
      pos_embed                      → pos_embed (if shapes match)
      norm                           → norm
    """
    n_transferred = 0

    # Top-level: proj, pos_embed, norm
    mapping_top = {
        'proj.weight': 'input_proj.weight',
        'proj.bias': 'input_proj.bias',
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
    }
    pre_sd = dict(pretrained_decoder.named_parameters())
    diff_sd = dict(diff_decoder.named_parameters())

    # pos_embed (buffer, not parameter)
    pre_pos = pretrained_decoder.pos_embed
    diff_pos = diff_decoder.pos_embed
    if pre_pos.shape == diff_pos.shape:
        diff_pos.data.copy_(pre_pos.data)
        n_transferred += 1
        if verbose:
            print(f"  pos_embed: {list(pre_pos.shape)}")

    for pre_name, diff_name in mapping_top.items():
        if pre_name in pre_sd and diff_name in diff_sd:
            pre_p = pre_sd[pre_name]
            diff_p = diff_sd[diff_name]
            if pre_p.shape == diff_p.shape:
                diff_p.data.copy_(pre_p.data)
                n_transferred += 1
                if verbose:
                    print(f"  {pre_name} → {diff_name}: {list(pre_p.shape)}")

    # Per-block transfer
    n_blocks = min(len(pretrained_decoder.blocks), len(diff_decoder.blocks))
    for i in range(n_blocks):
        pre_block = pretrained_decoder.blocks[i]
        diff_block = diff_decoder.blocks[i]

        block_mapping = {
            # Self-attention
            'attn.qkv.weight': 'self_attn.in_proj_weight',
            'attn.qkv.bias': 'self_attn.in_proj_bias',
            'attn.proj.weight': 'self_attn.out_proj.weight',
            'attn.proj.bias': 'self_attn.out_proj.bias',
            # Norms
            'norm1.weight': 'norm1.weight',
            'norm1.bias': 'norm1.bias',
            'norm2.weight': 'norm3.weight',  # pre-MLP norm
            'norm2.bias': 'norm3.bias',
            # MLP
            'mlp.fc1.weight': 'mlp.0.weight',
            'mlp.fc1.bias': 'mlp.0.bias',
            'mlp.fc2.weight': 'mlp.2.weight',
            'mlp.fc2.bias': 'mlp.2.bias',
        }

        pre_block_sd = dict(pre_block.named_parameters())
        diff_block_sd = dict(diff_block.named_parameters())

        for pre_name, diff_name in block_mapping.items():
            if pre_name in pre_block_sd and diff_name in diff_block_sd:
                pre_p = pre_block_sd[pre_name]
                diff_p = diff_block_sd[diff_name]
                if pre_p.shape == diff_p.shape:
                    diff_p.data.copy_(pre_p.data)
                    n_transferred += 1

        if verbose:
            print(f"  block {i}: transferred attention + MLP + norms")

    return n_transferred


def main():
    args = get_args()
    cfg = load_config(args.config)

    rank, world_size, local_rank, distributed = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    train_cfg = cfg['training']
    model_cfg = cfg['model']
    diff_cfg = cfg['diffusion']

    # Load encoder
    pretrained = SubDiff(
        img_size=cfg['data']['image_size'],
        patch_size=model_cfg['encoder_patch_size'],
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        decoder_dim=model_cfg.get('pretrain_decoder_dim', 512),
        decoder_depth=model_cfg.get('pretrain_decoder_depth', 4),
        decoder_num_heads=model_cfg.get('pretrain_decoder_num_heads', 8),
        total_epochs=1,
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        pretrained.load_state_dict(ckpt['model'])
        if is_main:
            print(f"Loaded pretrained encoder from epoch {ckpt['epoch']}")
    else:
        if is_main:
            print("No checkpoint provided — training from scratch (baseline)")

    encoder = pretrained.get_encoder()

    # Build diffusion finetune model
    model = DiffusionFinetune(
        encoder=encoder,
        encoder_dim=model_cfg['embed_dim'],
        encoder_patch_size=model_cfg['encoder_patch_size'],
        decoder_patch_size=model_cfg['decoder_patch_size'],
        img_size=cfg['data']['image_size'],
        decoder_dim=model_cfg['decoder_dim'],
        decoder_depth=model_cfg['decoder_depth'],
        decoder_num_heads=model_cfg['decoder_num_heads'],
        num_timesteps=diff_cfg['num_timesteps'],
        beta_start=diff_cfg['beta_start'],
        beta_end=diff_cfg['beta_end'],
        schedule_type=diff_cfg['schedule_type'],
        freeze_encoder=train_cfg.get('freeze_encoder', True),
    ).to(device)

    # Transfer pretrained decoder weights to diffusion decoder
    if args.checkpoint and train_cfg.get('init_decoder_from_pretrained', False):
        pretrained_decoder = pretrained.decoder
        diff_decoder = model.decoder
        n_transfer = _transfer_decoder_weights(pretrained_decoder, diff_decoder, is_main)
        if is_main:
            print(f"Transferred {n_transfer} tensors from pretrained decoder to diffusion decoder")

    if is_main:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Diffusion model: {total_params:.1f}M params, {trainable:.1f}M trainable")
        print(f"Encoder patch: {model_cfg['encoder_patch_size']}x{model_cfg['encoder_patch_size']}, "
              f"Decoder patch: {model_cfg['decoder_patch_size']}x{model_cfg['decoder_patch_size']}")

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    model_without_ddp = model.module if distributed else model

    # Data
    train_loader, train_sampler = build_pretrain_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=train_cfg['batch_size'],
        num_workers=cfg['data']['num_workers'],
        distributed=distributed,
    )

    # Optimizer (only decoder params if encoder is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        betas=(0.9, 0.999),
    )

    # Tensorboard
    writer = None
    if is_main:
        log_dir = cfg['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    # Training loop
    total_epochs = train_cfg['epochs']

    for epoch in range(total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        cur_lr = cosine_lr_schedule(
            optimizer, epoch, total_epochs,
            train_cfg['warmup_epochs'], train_cfg['lr'], train_cfg['min_lr'],
        )

        model.train()
        total_loss = 0.0
        num_steps = 0

        for step, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)

            loss, log_dict = model_without_ddp.forward(imgs) if not distributed \
                else model(imgs)

            optimizer.zero_grad()
            loss.backward()
            if train_cfg.get('clip_grad', 0) > 0:
                nn.utils.clip_grad_norm_(trainable_params, train_cfg['clip_grad'])
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1
            global_step = epoch * len(train_loader) + step

            if is_main and step % cfg['logging']['print_every'] == 0:
                print(f"Epoch [{epoch}/{total_epochs}] Step [{step}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={cur_lr:.6f}")

            if writer and step % cfg['logging']['print_every'] == 0:
                writer.add_scalar('diffusion/loss', loss.item(), global_step)
                writer.add_scalar('diffusion/lr', cur_lr, global_step)

        avg_loss = total_loss / max(num_steps, 1)
        if is_main:
            print(f"Epoch [{epoch}/{total_epochs}] avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if is_main and (epoch + 1) % cfg['logging']['save_every'] == 0:
            ckpt_dir = os.path.join(cfg['logging']['log_dir'], 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(ckpt_dir, f'finetune_diff_{epoch:04d}.pth'))

    # Save final
    if is_main:
        ckpt_dir = os.path.join(cfg['logging']['log_dir'], 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': total_epochs - 1,
        }, os.path.join(ckpt_dir, 'finetune_diff_final.pth'))
        print("Saved final diffusion checkpoint")

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
