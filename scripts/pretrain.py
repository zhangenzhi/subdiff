"""
SubDiff pretraining script with DDP support.

Usage:
  Single GPU:
    python scripts/pretrain.py --config configs/pretrain_vit_b16.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_vit_b16.yaml
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff
from subdiff.data import build_pretrain_dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr_schedule(optimizer, epoch, total_epochs, warmup_epochs, lr, min_lr):
    if epoch < warmup_epochs:
        cur_lr = lr * epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cur_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def main():
    args = get_args()
    cfg = load_config(args.config)

    rank, world_size, local_rank, distributed = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Build model
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

    if is_main:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"SubDiff model: {param_count:.1f}M parameters")
        print(f"Curriculum: {model.curriculum}")

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    model_without_ddp = model.module if distributed else model

    # Build dataloader
    train_loader, train_sampler = build_pretrain_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        distributed=distributed,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
        betas=(0.9, 0.95),
    )

    # Tensorboard
    writer = None
    if is_main:
        os.makedirs(cfg['logging']['log_dir'], exist_ok=True)
        writer = SummaryWriter(cfg['logging']['log_dir'])

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    total_epochs = cfg['training']['epochs']
    print_every = cfg['logging']['print_every']
    save_every = cfg['logging']['save_every']

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        cur_lr = cosine_lr_schedule(
            optimizer, epoch, total_epochs,
            cfg['training']['warmup_epochs'],
            cfg['training']['lr'],
            cfg['training']['min_lr'],
        )

        # Get curriculum state for logging
        curriculum_state = model_without_ddp.curriculum.get_state(epoch)

        model.train()
        total_loss = 0.0
        num_steps = 0

        for step, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)

            loss, log_dict = model_without_ddp.forward(imgs, epoch=epoch) \
                if not distributed else model.module.forward(imgs, epoch=epoch)

            # For DDP: need to compute loss through DDP wrapper for gradient sync
            if distributed:
                # Re-forward through DDP for proper gradient sync
                loss, log_dict = _forward_ddp(model, imgs, epoch)

            optimizer.zero_grad()
            loss.backward()

            if cfg['training']['clip_grad'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad'])

            optimizer.step()

            total_loss += loss.item()
            num_steps += 1
            global_step = epoch * len(train_loader) + step

            if is_main and step % print_every == 0:
                print(f"Epoch [{epoch}/{total_epochs}] Step [{step}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} noisy_loss={log_dict['noisy_loss']:.4f} "
                      f"clean_loss={log_dict['clean_loss']:.4f} "
                      f"t=[{curriculum_state['t_min']},{curriculum_state['t_max']}] "
                      f"clean_ratio={curriculum_state['clean_ratio']:.3f} lr={cur_lr:.6f}")

            if writer and step % print_every == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/noisy_loss', log_dict['noisy_loss'], global_step)
                writer.add_scalar('train/clean_loss', log_dict['clean_loss'], global_step)
                writer.add_scalar('train/lr', cur_lr, global_step)
                writer.add_scalar('curriculum/t_min', curriculum_state['t_min'], global_step)
                writer.add_scalar('curriculum/t_max', curriculum_state['t_max'], global_step)
                writer.add_scalar('curriculum/clean_ratio', curriculum_state['clean_ratio'], global_step)
                writer.add_scalar('curriculum/t_mean', log_dict['t_mean'], global_step)

        avg_loss = total_loss / max(num_steps, 1)
        if is_main:
            print(f"Epoch [{epoch}/{total_epochs}] avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if is_main and (epoch + 1) % save_every == 0:
            ckpt_dir = os.path.join(cfg['logging']['log_dir'], 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': cfg,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final
    if is_main:
        ckpt_dir = os.path.join(cfg['logging']['log_dir'], 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        final_path = os.path.join(ckpt_dir, 'checkpoint_final.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': total_epochs - 1,
            'config': cfg,
        }, final_path)
        print(f"Saved final checkpoint: {final_path}")

    if distributed:
        dist.destroy_process_group()


def _forward_ddp(model, imgs, epoch):
    """Forward pass through DDP model.

    We override the forward to pass epoch through. The DDP wrapper calls
    model.module.forward, so we temporarily store epoch.
    """
    model.module._current_epoch = epoch
    # Use a simple wrapper
    loss, log_dict = _SubDiffDDPForward.apply(model, imgs, epoch)
    return loss, log_dict


class _SubDiffDDPForward:
    """Helper to run forward through DDP for gradient sync."""

    @staticmethod
    def apply(model, imgs, epoch):
        # DDP needs the forward to go through the wrapper
        # We use a hook approach: the model.forward calls module.forward
        # Store epoch so module can access it
        model.module._ddp_epoch = epoch
        # Patch forward temporarily
        original_forward = model.module.forward

        def patched_forward(x):
            return original_forward(x, epoch=model.module._ddp_epoch)

        model.module.forward = patched_forward
        loss, log_dict = model(imgs)
        model.module.forward = original_forward
        return loss, log_dict


if __name__ == '__main__':
    main()
