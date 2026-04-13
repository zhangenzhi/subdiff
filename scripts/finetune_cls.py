"""
Classification finetuning for SubDiff pretrained encoder on ImageNet.

Usage:
  Single GPU:
    python scripts/finetune_cls.py --config configs/finetune_cls.yaml \
        --checkpoint logs/checkpoints/checkpoint_final.pth

  Multi-GPU:
    torchrun --nproc_per_node=8 scripts/finetune_cls.py \
        --config configs/finetune_cls.yaml \
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff
from subdiff.data import build_pretrain_dataloader, build_eval_dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='pretrained checkpoint')
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
        cur_lr = lr * epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cur_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


class ClassificationModel(nn.Module):
    """Encoder + classification head."""

    def __init__(self, encoder, embed_dim, num_classes=1000):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        cls_token, _ = self.encoder(x)
        return self.head(cls_token)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    correct_top5 = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        # Top-1
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        # Top-5
        _, top5_preds = logits.topk(5, dim=-1)
        correct_top5 += (top5_preds == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        total += labels.shape[0]
    return correct / total, correct_top5 / total


def build_param_groups(model, lr, layer_decay, encoder_depth):
    """Build parameter groups with layer-wise lr decay for the encoder."""
    param_groups = []

    # Classification head: full lr
    param_groups.append({
        'params': list(model.head.parameters()),
        'lr': lr,
        'name': 'head',
    })

    # Encoder: layer-wise lr decay
    encoder = model.encoder

    # Patch embed and pos embed: deepest decay
    no_decay_params = []
    decay_params = []
    for name, param in encoder.named_parameters():
        if 'blocks' in name:
            continue  # handled per-layer below
        if 'bias' in name or 'norm' in name or 'pos_embed' in name or 'cls_token' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    scale = layer_decay ** encoder_depth
    if decay_params:
        param_groups.append({
            'params': decay_params,
            'lr': lr * scale,
            'weight_decay': 0.05,
            'name': 'encoder_embed',
        })
    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'lr': lr * scale,
            'weight_decay': 0.0,
            'name': 'encoder_embed_no_decay',
        })

    # Transformer blocks: per-layer decay
    for i, block in enumerate(encoder.blocks):
        block_decay_params = []
        block_no_decay_params = []
        for name, param in block.named_parameters():
            if 'bias' in name or 'norm' in name:
                block_no_decay_params.append(param)
            else:
                block_decay_params.append(param)

        block_scale = layer_decay ** (encoder_depth - i)
        if block_decay_params:
            param_groups.append({
                'params': block_decay_params,
                'lr': lr * block_scale,
                'weight_decay': 0.05,
                'name': f'encoder_block_{i}',
            })
        if block_no_decay_params:
            param_groups.append({
                'params': block_no_decay_params,
                'lr': lr * block_scale,
                'weight_decay': 0.0,
                'name': f'encoder_block_{i}_no_decay',
            })

    return param_groups


def main():
    args = get_args()
    cfg = load_config(args.config)

    rank, world_size, local_rank, distributed = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    train_cfg = cfg['training']
    model_cfg = cfg['model']

    # Load pretrained encoder
    pretrained = SubDiff(
        img_size=cfg['data']['image_size'],
        patch_size=model_cfg['patch_size'],
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        decoder_dim=model_cfg.get('decoder_embed_dim', 512),
        decoder_depth=model_cfg.get('decoder_depth', 4),
        decoder_num_heads=model_cfg.get('decoder_num_heads', 8),
        total_epochs=1,  # dummy
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    pretrained.load_state_dict(ckpt['model'])
    if is_main:
        print(f"Loaded pretrained checkpoint from epoch {ckpt['epoch']}")

    encoder = pretrained.get_encoder()
    model = ClassificationModel(
        encoder, model_cfg['embed_dim'], train_cfg['num_classes']
    ).to(device)

    if is_main:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Classification model: {param_count:.1f}M params, {trainable:.1f}M trainable")

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
    val_loader, _ = build_eval_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=train_cfg['batch_size'],
        num_workers=cfg['data']['num_workers'],
        distributed=distributed,
    )

    # Optimizer with layer-wise lr decay
    param_groups = build_param_groups(
        model_without_ddp,
        lr=train_cfg['lr'],
        layer_decay=train_cfg['layer_decay'],
        encoder_depth=model_cfg['depth'],
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # Mixup / label smoothing
    label_smoothing = train_cfg.get('label_smoothing', 0.1)
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Tensorboard
    writer = None
    if is_main:
        log_dir = cfg['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    total_epochs = train_cfg['epochs']
    best_acc = 0.0

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

        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if train_cfg.get('clip_grad', 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg['clip_grad'])
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

            if is_main and step % cfg['logging']['print_every'] == 0:
                print(f"Epoch [{epoch}/{total_epochs}] Step [{step}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={cur_lr:.6f}")

        avg_loss = total_loss / max(num_steps, 1)

        # Evaluate
        top1, top5 = evaluate(model, val_loader, device)
        best_acc = max(best_acc, top1)

        if is_main:
            print(f"Epoch [{epoch}/{total_epochs}] avg_loss={avg_loss:.4f} "
                  f"top1={top1:.4f} top5={top5:.4f} best={best_acc:.4f}")
            if writer:
                writer.add_scalar('finetune/loss', avg_loss, epoch)
                writer.add_scalar('finetune/top1', top1, epoch)
                writer.add_scalar('finetune/top5', top5, epoch)
                writer.add_scalar('finetune/lr', cur_lr, epoch)

        # Save checkpoint
        if is_main and (epoch + 1) % cfg['logging']['save_every'] == 0:
            ckpt_dir = os.path.join(cfg['logging']['log_dir'], 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'top1': top1,
                'best_acc': best_acc,
            }, os.path.join(ckpt_dir, f'finetune_cls_{epoch:04d}.pth'))

    if is_main:
        print(f"\nBest top-1 accuracy: {best_acc:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
