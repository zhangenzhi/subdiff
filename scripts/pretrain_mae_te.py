"""Run 12e Stage 1: pure MAE training with TE fp8.

Trains a μ-generator at p=4 on ImageNet using:
  - subdiff/model_te.py MAEViT (12 te.TransformerLayer blocks)
  - te.fp8_autocast around forward+backward
  - DDP, find_unused_parameters=False
  - bf16 params, fp8 GEMMs, fp32 master weights via AdamW

Output checkpoint serves as μ-generator for Stage 2 cold-RF Refiner.

Usage (sdif env, with cuda module loaded):
  torchrun --nproc_per_node=N scripts/pretrain_mae_te.py \\
    --config configs/pretrain_mae_p4_fp8.yaml
"""

import os
import sys
import argparse
import math
import time
import yaml

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model_te import MAEClassicViT
from subdiff.data import build_pretrain_dataloader


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()


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


def cosine_lr(opt, epoch, total, warmup, lr, min_lr, schedule):
    if schedule == 'constant':
        cur = lr
    elif epoch < warmup:
        cur = lr * (epoch + 1) / max(warmup, 1)
    else:
        prog = (epoch - warmup) / max(total - warmup, 1)
        cur = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
    for g in opt.param_groups:
        g['lr'] = cur
    return cur


def _enable_hpc_speedups():
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Use cuDNN attention (FA3 on Hopper) — TE TransformerLayer also benefits
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)


def main():
    args = get_args()
    cfg = load_config(args.config)

    _enable_hpc_speedups()

    rank, world_size, local_rank, distributed = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')

    if is_main:
        print(f"[HPC] PyTorch {torch.__version__}, "
              f"GPU={torch.cuda.get_device_name(0)}, "
              f"world_size={world_size}")

    # Build classical asymmetric MAE. Cast everything to bf16 to match TE
    # TransformerLayer's params_dtype=bf16 — otherwise patch_embed/pos_embed
    # (fp32 by default) feed fp32 into TE blocks which expect bf16 input
    # under no-autocast.
    model = MAEClassicViT(
        img_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        in_chans=3,
        embed_dim=cfg['model']['embed_dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        decoder_dim=cfg['model']['decoder_dim'],
        decoder_depth=cfg['model']['decoder_depth'],
        decoder_num_heads=cfg['model']['decoder_num_heads'],
        clean_ratio=cfg['model']['clean_ratio'],
        params_dtype=torch.bfloat16,
    ).to(device).bfloat16()

    if is_main:
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"MAEViT: {n:.1f}M params, p={cfg['model']['patch_size']}, "
              f"N={model.num_patches}, clean_ratio={model.clean_ratio}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model_raw = model.module if distributed else model

    # Data
    train_loader, train_sampler = build_pretrain_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        distributed=distributed,
        backend=cfg.get('data', {}).get('backend', 'torch'),
        transform_type=cfg.get('data', {}).get('transform_type', 'diffusion'),
    )

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
        betas=(0.9, 0.95),
    )

    # fp8 recipe (HYBRID: E4M3 fwd, E5M2 bwd — NVIDIA-recommended)
    fp8_enabled = cfg['training'].get('fp8', True)
    fp8_recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID)
    if is_main:
        print(f"fp8: {fp8_enabled}  recipe=DelayedScaling(margin=0, HYBRID)")

    # Tensorboard + ckpt dir
    writer = None
    if is_main:
        os.makedirs(cfg['logging']['log_dir'], exist_ok=True)
        writer = SummaryWriter(cfg['logging']['log_dir'])

    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model_raw.load_state_dict(ckpt['model'], strict=False)
        try:
            opt.load_state_dict(ckpt['optimizer'])
        except (ValueError, KeyError):
            if is_main:
                print("WARN: optimizer state not restored")
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        if is_main:
            print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    total_epochs = cfg['training']['epochs']
    bs_per_gpu = cfg['training']['batch_size']
    eff_bs = bs_per_gpu * world_size
    print_every = cfg['logging']['print_every']
    save_every = cfg['logging']['save_every']

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        cur_lr = cosine_lr(opt, epoch, total_epochs,
                           cfg['training']['warmup_epochs'],
                           cfg['training']['lr'],
                           cfg['training']['min_lr'],
                           cfg['training'].get('lr_schedule', 'cosine'))

        model.train()
        total_loss = 0.0
        n_steps = 0
        epoch_start = time.time()
        last_log = time.time()
        last_log_step = 0
        t_iter = time.time()

        for step, (imgs, _) in enumerate(train_loader):
            t_data = time.time() - t_iter
            # Cast to bf16 to match model params (probe pattern that worked).
            # te.fp8_autocast handles TE blocks; non-TE ops (Conv2d patch_embed,
            # nn.Linear head) need bf16 input directly.
            imgs = imgs.to(device, non_blocking=True).bfloat16()

            with te.fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe):
                loss, log = model(imgs)

            opt.zero_grad()
            loss.backward()
            if cfg['training'].get('clip_grad', 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(),
                                         cfg['training']['clip_grad'])
            opt.step()

            total_loss += loss.item()
            n_steps += 1
            global_step = epoch * len(train_loader) + step
            t_step = time.time() - t_iter

            if is_main and step % print_every == 0:
                now = time.time()
                step_per_s = max(step - last_log_step, 1) / max(now - last_log, 1e-6)
                imgs_per_s = step_per_s * eff_bs
                util = 1.0 - (t_data / max(t_step, 1e-6))
                print(f"Epoch [{epoch}/{total_epochs}] Step [{step}/{len(train_loader)}] "
                      f"loss={log['loss'].item():.4f} "
                      f"noisy_loss={log['noisy_loss'].item():.4f} "
                      f"clean_loss={log['clean_loss'].item():.4f} "
                      f"clean_ratio={log['clean_ratio']:.3f} "
                      f"lr={cur_lr:.6f} "
                      f"| {step_per_s:.2f} step/s {imgs_per_s:.0f} img/s "
                      f"data={t_data*1000:.0f}ms step={t_step*1000:.0f}ms util={util:.0%}")
                last_log = now
                last_log_step = step

            if writer and step % print_every == 0:
                writer.add_scalar('train/loss', log['loss'].item(), global_step)
                writer.add_scalar('train/noisy_loss', log['noisy_loss'].item(), global_step)
                writer.add_scalar('train/clean_loss', log['clean_loss'].item(), global_step)
                writer.add_scalar('train/lr', cur_lr, global_step)
                writer.add_scalar('perf/step_ms', t_step * 1000, global_step)

            t_iter = time.time()

        avg_loss = total_loss / max(n_steps, 1)
        epoch_wall = time.time() - epoch_start
        if is_main:
            print(f"Epoch [{epoch}/{total_epochs}] avg_loss={avg_loss:.4f} "
                  f"| wall={epoch_wall:.0f}s ({epoch_wall/60:.1f}min) "
                  f"throughput={(n_steps*eff_bs)/max(epoch_wall,1e-6):.0f} img/s")
            if writer:
                writer.add_scalar('perf/epoch_wall_sec', epoch_wall, epoch)
                writer.add_scalar('train/avg_loss', avg_loss, epoch)

        # Save checkpoints
        if is_main and (epoch + 1) % save_every == 0:
            ckpt_dir = os.path.join(cfg['logging']['log_dir'], 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            state = {
                'model': model_raw.state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': epoch,
                'avg_loss': avg_loss,
                'best_loss': best_loss,
                'config': cfg,
            }
            latest = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
            torch.save(state, latest)
            print(f"Saved latest: {latest} (avg_loss={avg_loss:.4f})")
            if is_best:
                best = os.path.join(ckpt_dir, 'checkpoint_best.pth')
                torch.save(state, best)
                print(f"Saved best: {best} (avg_loss={avg_loss:.4f})")

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
