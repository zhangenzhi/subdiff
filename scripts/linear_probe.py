"""
Linear probe evaluation for SubDiff pretrained encoder.

Freezes the encoder and trains a linear classifier on top of the CLS token.

Usage:
    python scripts/linear_probe.py \
        --config configs/pretrain_vit_b16.yaml \
        --checkpoint logs/checkpoints/checkpoint_final.pth
"""

import os
import sys
import argparse
import yaml

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff
from subdiff.data import build_eval_dataloader, build_pretrain_dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--probe_lr', type=float, default=None)
    parser.add_argument('--probe_epochs', type=int, default=None)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class LinearProbe(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes=1000):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            cls_token, _ = self.encoder(x)
        return self.head(cls_token)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return correct / total


def main():
    args = get_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    probe_lr = args.probe_lr or cfg['probe']['probe_lr']
    probe_epochs = args.probe_epochs or cfg['probe']['probe_epochs']
    num_classes = cfg['probe']['num_classes']

    # Load pretrained model
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
    )

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    encoder = model.get_encoder().to(device)
    probe = LinearProbe(encoder, cfg['model']['embed_dim'], num_classes).to(device)

    # Data
    train_loader, _ = build_pretrain_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data']['num_workers'],
    )
    val_loader, _ = build_eval_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data']['num_workers'],
    )

    # Optimizer for linear head only
    optimizer = torch.optim.SGD(
        probe.head.parameters(),
        lr=probe_lr,
        momentum=0.9,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, probe_epochs)
    criterion = nn.CrossEntropyLoss()

    # Train linear probe
    best_acc = 0.0
    for epoch in range(probe_epochs):
        probe.train()
        total_loss = 0.0
        num_steps = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = probe(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

        scheduler.step()
        avg_loss = total_loss / max(num_steps, 1)

        # Evaluate
        val_acc = evaluate(probe, val_loader, device)
        best_acc = max(best_acc, val_acc)
        print(f"Probe Epoch [{epoch+1}/{probe_epochs}] "
              f"loss={avg_loss:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}")

    print(f"\nLinear probe result: {best_acc:.4f}")
    return best_acc


if __name__ == '__main__':
    main()
