"""
ImageNet data loading for SubDiff pretraining and linear probe evaluation.
"""

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def build_pretrain_transform(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_pretrain_dataloader(imagenet_dir, image_size=224, batch_size=256,
                              num_workers=8, distributed=False):
    train_dir = os.path.join(imagenet_dir, 'train')
    transform = build_pretrain_transform(image_size)
    dataset = datasets.ImageFolder(train_dir, transform=transform)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def build_eval_dataloader(imagenet_dir, image_size=224, batch_size=256,
                          num_workers=8, split='val', distributed=False):
    data_dir = os.path.join(imagenet_dir, split)
    transform = build_eval_transform(image_size)
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, sampler
